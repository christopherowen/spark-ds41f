# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from collections.abc import Mapping, Sequence
import os
from typing import Any

import torch

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    DraftBlockTableLayout,
)
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    markov_anchor_embed,
    register_greedy_markov_shard_pair_warmup,
    register_sample_greedy_markov_warmup,
    register_select_first_argmax_warmup,
    sample_greedy_markov,
    scratch_shape,
    select_first_argmax,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

_MAX_STEPS_BY_BATCH_ENV = os.environ.get("SPARK_DS41F_DSPARK_MAX_STEPS_BY_BATCH", "")


def parse_max_steps_by_batch(
    spec: str | Mapping[int, int] | Sequence[tuple[int, int]] | None,
    num_speculative_steps: int,
) -> tuple[tuple[int, int], ...]:
    """Parse concurrency-bounded speculative step schedule ``(min_reqs, max_steps)``.

    Entries are sorted by ``min_reqs`` ascending, clamped to
    ``[1, num_speculative_steps]``, and made monotonically non-increasing in
    ``max_steps`` as ``min_reqs`` increases.
    """
    if not spec or num_speculative_steps <= 0:
        return ()
    raw_pairs: list[tuple[int, int]] = []
    if isinstance(spec, str):
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"Invalid max_steps_by_batch entry {token!r}; expected 'batch:steps'"
                )
            batch_s, steps_s = token.split(":", 1)
            raw_pairs.append((int(batch_s.strip()), int(steps_s.strip())))
    elif isinstance(spec, Mapping):
        for batch_k, steps_v in spec.items():
            raw_pairs.append((int(batch_k), int(steps_v)))
    else:
        for batch_k, steps_v in spec:
            raw_pairs.append((int(batch_k), int(steps_v)))

    by_batch: dict[int, int] = {}
    for min_reqs, max_steps in sorted(raw_pairs, key=lambda item: item[0]):
        if min_reqs < 1:
            raise ValueError(f"min_reqs must be >= 1, got {min_reqs}")
        clamped = max(1, min(int(max_steps), int(num_speculative_steps)))
        by_batch[min_reqs] = min(by_batch.get(min_reqs, clamped), clamped)

    normalized: list[tuple[int, int]] = []
    running_cap = int(num_speculative_steps)
    for min_reqs in sorted(by_batch):
        running_cap = min(running_cap, by_batch[min_reqs])
        if running_cap < num_speculative_steps and (
            not normalized or normalized[-1][1] != running_cap
        ):
            normalized.append((min_reqs, running_cap))
    return tuple(normalized)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False
        draft_vocab = max(
            self.draft_model_config.hf_config.vocab_size,
            getattr(self.draft_model_config.hf_config, "draft_vocab_size", None) or 0,
        )
        shape = scratch_shape(self.max_num_reqs, draft_vocab)
        self._greedy_partial_values = torch.empty(
            shape, dtype=torch.float32, device=device
        )
        self._greedy_partial_indices = torch.empty(
            shape, dtype=torch.int32, device=device
        )
        # Per-request (max, global index) of this rank's vocabulary shard,
        # for vocabulary-parallel greedy drafts.
        # (value, index) and two unused columns: 16-byte rows, which the
        # one-shot all-gather writes in place (no staging or reshape copies).
        self._greedy_shard_pairs = torch.empty(
            (self.max_num_reqs, 4), dtype=torch.float32, device=device
        )
        # The vocabulary-parallel greedy steps' Markov embeddings, which the
        # anchor lookup and each step's selection write (with a replicated BF16
        # W1): the next step's rows, and every step's in the (request, step)
        # order the confidence head reads.
        rank = getattr(self.draft_model_config.hf_config, "dspark_markov_rank", 0)
        self._markov_step_embed = torch.empty(
            (self.max_num_reqs, rank), dtype=torch.bfloat16, device=device
        )
        self._markov_confidence_embeds = torch.empty(
            (self.max_num_reqs, self.num_speculative_steps, rank),
            dtype=torch.bfloat16,
            device=device,
        )
        schedule_spec = _MAX_STEPS_BY_BATCH_ENV or getattr(
            self.draft_model_config.hf_config, "dspark_max_steps_by_batch", None
        )
        self._max_steps_by_batch = parse_max_steps_by_batch(
            schedule_spec, self.num_speculative_steps
        )
        self._active_sample_indices = torch.empty(
            self.max_num_reqs * self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self._active_confidence_out = torch.empty(
            self.max_num_reqs * self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        if self._max_steps_by_batch:
            logger.info(
                "DSpark concurrency-bounded draft depth schedule: %s (max=%d)",
                self._max_steps_by_batch,
                self.num_speculative_steps,
            )

    def active_speculative_steps(self, num_reqs: int) -> int:
        """Return active sequential Markov draft steps for batch size ``num_reqs``."""
        n_spec = self.num_speculative_steps
        schedule = getattr(self, "_max_steps_by_batch", ())
        if not schedule or num_reqs <= 0:
            return n_spec
        steps = n_spec
        for min_reqs, max_steps in schedule:
            if num_reqs >= min_reqs:
                steps = min(steps, max_steps)
            else:
                break
        return max(1, min(steps, n_spec))

    def _configured_step_counts(self) -> tuple[int, ...]:
        """All distinct active speculative step counts across batch sizes."""
        n_spec = self.num_speculative_steps
        schedule = getattr(self, "_max_steps_by_batch", ())
        if not schedule:
            return (n_spec,)
        counts = [n_spec]
        for _, steps in schedule:
            if 1 <= steps <= n_spec and steps not in counts:
                counts.append(steps)
        return tuple(counts)

    @property
    def attn_vllm_config(self) -> VllmConfig:
        config = super().attn_vllm_config
        if self.draft_model_config.hf_config.model_type != "glm53_dspark":
            return config
        # The draft's sliding-window MLA geometry differs from the GLM verifier.
        config.model_config = self.draft_model_config
        config.quant_config = None
        config.attention_config = replace(
            config.attention_config,
            backend=self.speculative_config.attention_backend,
        )
        if self.speculative_config.kv_cache_dtype is not None:
            config.cache_config = replace(
                config.cache_config,
                cache_dtype=self.speculative_config.kv_cache_dtype,
            )
        return config

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        bind_auxiliary_stream = getattr(model, "bind_target_auxiliary_stream", None)
        if callable(bind_auxiliary_stream):
            bind_auxiliary_stream(target_model, self.hidden_states)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        self.use_confidence_head = (
            self.enable_adaptive_verification
            and getattr(model.model, "confidence_head", None) is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampled = gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )
        if self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(
                logits,
                sampled,
                idx_map,
                self.temperature,
            )
        return sampled

    def _vocab_parallel_greedy(self, local_head: bool) -> bool:
        """Whether greedy drafts over the full target vocabulary stay sharded:
        each rank reduces its shard and only (max, index) pairs are gathered."""
        return (
            not local_head
            and getattr(self.model, "vocab_parallel_greedy", False)
            and self.draft_logits is None
            and self.model.draft_id_to_target_id is None
            and self.acceptance_estimator is None
        )

    def register_warmups(
        self, block_table_layouts: Sequence[DraftBlockTableLayout]
    ) -> None:
        """Also register the greedy reduction of each draft position: the
        vocabulary-parallel one, whose base logits and Markov bias are this rank's
        shards, written in the draft head's dtype (draft position ``i`` is column
        ``i`` of both the base logits' ``[requests, positions, shard]`` view and
        the draft tokens); or the full-vocabulary one, when no logits cache,
        vocabulary map or estimator needs the logits."""
        super().register_warmups(block_table_layouts)
        local_head = getattr(self.model, "supports_local_draft_argmax", lambda: False)()
        if self._draft_topk is not None:
            return
        if not self._vocab_parallel_greedy(local_head):
            processor = getattr(self.model, "logits_processor", None)
            if (
                not local_head
                and processor is not None
                and self.draft_logits is None
                and self.model.draft_id_to_target_id is None
                and getattr(self, "acceptance_estimator", None) is None
            ):
                register_sample_greedy_markov_warmup(
                    dtype=processor.head_dtype, output_dtype=self.draft_tokens.dtype
                )
            return
        vocab_start, shard_vocab = self.model.draft_vocab_shard
        dtype, head_width, markov_width = self.model.draft_logits_local_layout
        configured_steps = (
            self._configured_step_counts()
            if hasattr(self, "_configured_step_counts")
            else (self.num_speculative_steps,)
        )
        for steps in configured_steps:
            for i in range(steps):
                register_greedy_markov_shard_pair_warmup(
                    base_dtype=dtype,
                    base_stride=steps * head_width,
                    base_aligned=i * head_width * dtype.itemsize % 16 == 0,
                    bias_dtype=dtype,
                    bias_stride=markov_width,
                    vocab=shard_vocab,
                    vocab_start=vocab_start,
                )
                register_select_first_argmax_warmup(
                    ranks=get_tensor_model_parallel_world_size(),
                    output_dtype=self.draft_tokens.dtype,
                    output_stride=self.draft_tokens.stride(0),
                    output_aligned=i * self.draft_tokens.element_size() % 16 == 0,
                )
        tilelang_inputs.declare(
            lambda: tilelang_inputs.gather_hidden_rows_calls(
                self.hidden_states.dtype, self.hidden_states.shape[1]
            )
        )
        table = self._markov_table()
        if table is not None:
            from vllm.v1.worker.gpu.sample import tilelang_sampling

            ids_dtype = self.input_buffers.input_ids.dtype
            tilelang_sampling.declare(
                lambda: tilelang_sampling.markov_embed_calls(
                    ids_dtype,
                    self._anchor_idx.dtype,
                    self.draft_tokens.dtype,
                    get_tensor_model_parallel_world_size(),
                    table.shape[1],
                )
            )

    def _markov_table(self) -> torch.Tensor | None:
        """The Markov W1 the vocabulary-parallel greedy steps look up in their own
        kernels: the model's replicated BF16 table, or None (``markov_embed``)."""
        table = getattr(self.model, "markov_embed_table", lambda: None)()
        if (
            table is None
            or table.dtype != torch.bfloat16
            or table.shape[1] != self._markov_step_embed.shape[1]
        ):
            return None
        return table

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        active_steps_fn = getattr(self, "active_speculative_steps", None)
        active_steps = (
            active_steps_fn(num_reqs) if callable(active_steps_fn) else n_spec
        )
        num_sample = num_reqs * n_spec
        active_sample = num_reqs * active_steps
        if active_steps < n_spec:
            indices_2d = self.sample_indices[:num_sample].view(num_reqs, n_spec)[
                :, :active_steps
            ]
            active_buf = getattr(self, "_active_sample_indices", None)
            if active_buf is not None:
                sample_indices = active_buf[:active_sample]
                sample_indices.view(num_reqs, active_steps).copy_(indices_2d)
            else:
                sample_indices = indices_2d.contiguous().view(-1)
        else:
            sample_indices = self.sample_indices[:num_sample]
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = tilelang_inputs.gather_hidden_rows(head_hidden, sample_indices)
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        local_head = getattr(self.model, "supports_local_draft_argmax", lambda: False)()
        vocab_parallel = self._vocab_parallel_greedy(local_head)
        if local_head:
            base_logits = self.model.compute_local_draft_logits(sample_hidden)
        elif vocab_parallel:
            vocab_start, shard_vocab = self.model.draft_vocab_shard
            base_logits = self.model.compute_draft_logits_local(sample_hidden)
        else:
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, active_steps, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        # The vocabulary-parallel greedy steps embed it, and each step's selection
        # the next step's token, in their own kernels.
        table = self._markov_table() if vocab_parallel else None
        if table is not None:
            step_embed = self._markov_step_embed[:num_reqs]
            confidence_embeds = self._markov_confidence_embeds[:num_reqs]
            markov_anchor_embed(
                self.input_buffers.input_ids,
                self._anchor_idx[:num_reqs],
                table,
                step_embed,
                confidence_embeds[:, 0],
            )
        else:
            prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(active_steps):
            # Sequential stage: Markov bias from the previously sampled token.
            if table is not None:
                markov_embed = step_embed
            else:
                markov_embed = self.model.markov_embed(prev)
                if self.use_confidence_head:
                    confidence_markov_embeds.append(markov_embed)
            if local_head:
                bias = self.model.compute_local_markov_bias(markov_embed)
                if self.draft_logits is None and self.acceptance_estimator is None:
                    draft_sampled_i = self.model.sample_local_draft_logits(
                        base_logits[:, i], bias
                    )
                else:
                    logits_i = self.model.gather_local_draft_logits(
                        base_logits[:, i], bias
                    )
                    draft_sampled_i = self._sample_logits(
                        logits_i, idx_map[:, i], sample_pos[:, i], i
                    )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            if vocab_parallel:
                bias = self.model.markov_bias_local(markov_embed)
                pairs = self._greedy_shard_pairs[:num_reqs]
                greedy_markov_shard_pair(
                    base_logits[:, i, :shard_vocab],
                    bias[:, :shard_vocab],
                    vocab_start,
                    pairs,
                    self._greedy_partial_values,
                    self._greedy_partial_indices,
                )
                draft_sampled_i = self.draft_tokens[:num_reqs, i]
                select_first_argmax(
                    tensor_model_parallel_all_gather(pairs, dim=-1),
                    draft_sampled_i,
                    pair_width=4,
                    embed=(
                        (table, step_embed, confidence_embeds[:, i + 1])
                        if table is not None and i + 1 < active_steps
                        else None
                    ),
                )
                prev = draft_sampled_i
                continue
            bias = self.model.markov_bias(markov_embed)
            if (
                self.draft_logits is None
                and self.model.draft_id_to_target_id is None
                and base_logits.dtype == bias.dtype
                and self.acceptance_estimator is None
            ):
                draft_sampled_i = self.draft_tokens[:num_reqs, i]
                sample_greedy_markov(
                    base_logits[:, i],
                    bias,
                    draft_sampled_i,
                    self._greedy_partial_values,
                    self._greedy_partial_indices,
                )
            else:
                logits_i = base_logits[:, i] + bias
                draft_sampled_i = self._sample_logits(
                    logits_i, idx_map[:, i], sample_pos[:, i], i
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if active_steps < n_spec:
            self.draft_tokens[:num_reqs, active_steps:].zero_()

        if self.use_confidence_head and table is not None:
            if active_steps == n_spec:
                # Straight into the probabilities' rows (contiguous: [:num_reqs]).
                self.model.compute_confidence(
                    sample_hidden,
                    confidence_embeds.flatten(0, 1),
                    out=self.draft_token_confidence_probs[:num_reqs].view(-1),
                )
            else:
                conf_buf = getattr(self, "_active_confidence_out", None)
                conf_out = (
                    conf_buf[:active_sample]
                    if conf_buf is not None
                    else torch.empty(
                        active_sample,
                        dtype=self.draft_token_confidence_probs.dtype,
                        device=sample_hidden.device,
                    )
                )
                self.model.compute_confidence(
                    sample_hidden,
                    confidence_embeds[:, :active_steps].reshape(active_sample, -1),
                    out=conf_out,
                )
                self.draft_token_confidence_probs[:num_reqs, :active_steps] = (
                    conf_out.view(num_reqs, active_steps)
                )
                self.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()
        elif self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs, :active_steps] = (
                confidence.view(num_reqs, active_steps)
            )
            if active_steps < n_spec:
                self.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()
        elif (
            active_steps < n_spec
            and getattr(self, "draft_token_confidence_probs", None) is not None
        ):
            self.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        active_steps_fn = getattr(self, "active_speculative_steps", None)
        active_steps = (
            active_steps_fn(num_reqs) if callable(active_steps_fn) else n_spec
        )
        num_sample = num_reqs * n_spec
        active_sample = num_reqs * active_steps
        if active_steps < n_spec:
            indices_2d = self.sample_indices[:num_sample].view(num_reqs, n_spec)[
                :, :active_steps
            ]
            active_buf = getattr(self, "_active_sample_indices", None)
            if active_buf is not None:
                sample_indices = active_buf[:active_sample]
                sample_indices.view(num_reqs, active_steps).copy_(indices_2d)
            else:
                sample_indices = indices_2d.contiguous().view(-1)
        else:
            sample_indices = self.sample_indices[:num_sample]
        sample_hidden = tilelang_inputs.gather_hidden_rows(head_hidden, sample_indices)
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, active_steps, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(active_steps):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if active_steps < n_spec:
            self.draft_tokens[:num_reqs, active_steps:].zero_()

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs, :active_steps] = (
                confidence.view(num_reqs, active_steps)
            )
            if active_steps < n_spec:
                self.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()
        elif (
            active_steps < n_spec
            and getattr(self, "draft_token_confidence_probs", None) is not None
        ):
            self.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
