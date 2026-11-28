# SPDX-License-Identifier: Apache-2.0
"""DSpark Markov and confidence heads with display carve-out and packed BF16 support."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab,
    ``vocab_size``); ``markov_w2`` projects it to a draft-vocab bias
    (``draft_vocab_size``) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.

    Weights are replicated by default. An opt-in TP-sharded layout reduces
    draft residency and supports rank-local greedy sampling. Keeping W1
    replicated avoids an embedding all-reduce at each draft position.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
        lm_head_quantization: Literal["mxfp8", "nvfp4"] | None = None,
        vocab_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.shard_across_tp = envs.VLLM_DSPARK_SHARD_MARKOV_HEAD
        self.replicate_w1 = envs.VLLM_DSPARK_REPLICATE_MARKOV_W1
        if self.replicate_w1 and not self.shard_across_tp:
            raise ValueError("Replicated Markov W1 requires a TP-sharded Markov head")
        if self.shard_across_tp and retain_weight_for_gather:
            raise ValueError("Gathered Markov top-k requires replicated weights")
        if self.shard_across_tp and not self.replicate_w1:
            self.markov_w1 = VocabParallelEmbedding(
                vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
            )
        else:
            from spark_ds41.worker import display_carveout

            data = display_carveout.allocate(
                (vocab_size, markov_rank),
                torch.get_default_dtype(),
                torch.get_default_device(),
            )
            self.markov_w1 = nn.Embedding(
                vocab_size,
                markov_rank,
                _weight=data,
                _freeze=data is not None,
            )
            self.markov_w1._display_carveout_required = True
        # vocab_parallel shards the projection by vocabulary rows like the
        # LM head, for callers that reduce the draft argmax across shards.
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=not (self.shard_across_tp or vocab_parallel),
            lm_head_quantization=lm_head_quantization,
        )
        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)

    def local_bias(
        self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor
    ) -> torch.Tensor:
        """Return the unprocessed bias shard matching the target LM-head shard."""
        if not self.shard_across_tp:
            raise RuntimeError("Local Markov bias requires a TP-sharded head")
        return logits_processor._apply_head(self.markov_w2, markov_embed, None)

    def apply_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply the Markov bias only to selected rows of ``logits``.

        The caller initializes ``logits`` to ``-inf`` once for all draft
        positions. This method scatters the corrected candidate values into
        that dense buffer so the normal sampler sees the truncated proposal.
        """
        if self.markov_w2._retain_weight_for_gather and self.markov_w2.is_w4a16_nvfp4:
            # Use the fast Triton kernel for W4A16 NVFP4 weights with BF16
            # activations on CUDA. FP16 activations, non-CUDA accelerators, and
            # CPU use the slower dequantize/scatter fallback below.
            if (
                current_platform.is_cuda()
                and markov_embed.is_cuda
                and markov_embed.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and logits.dtype == torch.bfloat16
            ):
                nvfp4_gathered_bias(
                    markov_embed,
                    self.markov_w2._nvfp4_weight_for_gather,
                    self.markov_w2._nvfp4_weight_scale_for_gather,
                    self.markov_w2._nvfp4_weight_global_scale_for_gather,
                    values,
                    index,
                    logits,
                    scale,
                )
                return logits

        corrected = self.apply_bias_topk(markov_embed, values, index, scale)
        return logits.scatter_(1, index, corrected)

    def apply_bias_topk(
        self,
        markov_embed: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Return ``values + markov_bias`` ([num_reqs, k]) on the selected candidates
        without scattering into a dense vocabulary buffer."""
        if self.markov_w2._retain_weight_for_gather and self.markov_w2.is_w4a16_nvfp4:
            flat_index = index.reshape(-1)
            packed_weight = self.markov_w2._nvfp4_weight_for_gather.index_select(
                0, flat_index
            )
            weight_scale = self.markov_w2._nvfp4_weight_scale_for_gather.index_select(
                0, flat_index
            )
            weight = dequantize_to_dtype(
                packed_weight,
                weight_scale,
                self.markov_w2._nvfp4_weight_global_scale_for_gather,
                dtype=markov_embed.dtype,
                block_size=self.markov_w2._nvfp4_group_size_for_gather,
                swizzle=False,
            ).view(*index.shape, -1)
        else:
            weight = self.markov_w2.weight[index]
            if weight.dtype != markov_embed.dtype:
                raise RuntimeError(
                    "Gathered Markov projection requires unquantized weights or "
                    "retained ModelOpt W4A16 NVFP4 weights."
                )
        corrected = values.unsqueeze(-1).clone()
        corrected.baddbmm_(
            weight,
            markov_embed.unsqueeze(-1),
            beta=1.0,
            alpha=scale,
        )
        return corrected.squeeze(-1)


class DSparkConfidenceHead(nn.Module):
    """DSpark acceptance-confidence head."""

    tp_invariant: bool = False

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        bias: bool = False,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = (
            torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        ).float()
        return self.proj(x).squeeze(-1)

    def probability(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The acceptance probability, ``sigmoid`` of the head's logit (into
        ``out`` when given)."""
        return torch.sigmoid(self(hidden, markov_embed), out=out)
