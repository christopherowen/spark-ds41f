# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config import SpeculativeConfig, get_current_vllm_config
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
    triton_warmup_inputs,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
    logits_rows,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import (
    compute_topk_scores,
    register_topk_scores_warmup,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.sink import SinkAlternates, SinkSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _RESAMPLE_BLOCK_SIZE,
    register_rejection_sample_warmups,
    rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.tilelang_resampled import register_insert

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


def _flatten_sampled_warmup_inputs(*, sampled_stride: int) -> dict[str, Any]:
    int64 = TritonWarmupTensor(torch.int64)
    int32 = TritonWarmupTensor(torch.int32)
    return triton_warmup_inputs(
        _flatten_sampled_kernel,
        int64,
        int64,
        sampled_stride,
        int32,
        int32,
        grid=(1,),
        num_warps=1,
    )


# The reference ``tilelang_inputs.flatten_sampled_kernel`` is tested against; the
# rejection sampler no longer launches it.
_flatten_sampled_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_flatten_sampled_warmup_inputs
)(_flatten_sampled_kernel)


@dataclass
class _SinkChunk:
    """A verification chunk's sink flags, and the alternates of its last
    draws once verified (sink rejection)."""

    flags: torch.Tensor
    alternates: torch.Tensor | None = None


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
    ):
        self.sampler = sampler
        self.watermark_key = watermark_key
        # Sink rejection (set by the model runner): the drafter-row cut and
        # the alternates of each request's last draw.
        self.sink: SinkSampler | None = None
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        # The most drafts a request verifies (copy drafts widen DSpark's block),
        # and the drafter's block: a step with no copy verifies at most that many,
        # and block verification runs its kernels at that width.
        self.num_speculative_steps = spec_config.num_verification_tokens
        self.num_draft_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def register_warmups(
        self,
        *,
        logits: Sequence[tuple[torch.dtype, int]],
        processed_logits: Sequence[tuple[torch.dtype, int]],
        vocab_size: int,
        has_draft_logits: bool,
        dead_rows: bool = False,
    ) -> None:
        """Register verification of target logits given as ``(dtype, row stride)``
        pairs, raw (``logits``) or after sampling parameters (``processed_logits``),
        and the logprobs of verified tokens. Draft logits and watermarking are not
        covered by Triton warmup; the TileLang insertion is always declared. When
        the step can have dead rows (``dead_rows``), block verification declares its
        kernels with a dead-row cut (and refuses the configurations its TileLang
        kernels do not serve), and the other methods the insertion capped at it."""
        triton_only = (
            has_draft_logits
            or self.watermark_key is not None
            or self.sampler.use_fp64_gumbel
        )
        if dead_rows and self.use_block_verification and triton_only:
            raise ValueError(
                "Adaptive verification's dead rows (SPARK_DS41F_DSPARK_DEAD_ROWS_TAU) "
                "need block verification's TileLang kernels, which end each block at "
                "its cut: one-hot drafts (draft_sample_method greedy), FP32 draws and "
                "no watermark. Set the threshold to 0 or use standard verification."
            )
        limits = (
            (False, True) if dead_rows and not self.use_block_verification else (False,)
        )
        register_insert(
            triton.cdiv(vocab_size, _RESAMPLE_BLOCK_SIZE),
            self.sampler.use_fp64_gumbel,
            self.num_speculative_steps + 1,
            limits,
        )
        if has_draft_logits:
            register_insert(
                get_current_vllm_config().parallel_config.tensor_parallel_size,
                self.sampler.use_fp64_gumbel,
                self.num_speculative_steps + 1,
                limits,
            )
        if has_draft_logits or self.watermark_key is not None:
            return
        for steps in dict.fromkeys((self.num_speculative_steps, self.num_draft_steps)):
            register_rejection_sample_warmups(
                logits=(*logits, *processed_logits),
                vocab_size=vocab_size,
                num_speculative_steps=steps,
                synthetic_mode=self.synthetic_conditional_rates is not None,
                use_block_verification=self.use_block_verification,
                use_fp64=self.sampler.use_fp64_gumbel,
                dead_rows=dead_rows,
            )
        tilelang_inputs.declare(tilelang_inputs.flatten_sampled_calls)
        use_processed = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        register_topk_scores_warmup(
            logits=processed_logits if use_processed else logits,
            vocab_size=vocab_size,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        logprobs_dims: tuple[int, int] | None,
    ) -> LogprobsTensors | None:
        """Each logits row's scores, as the sampler's: the top logprobs, or
        the request's own ``logprob_token_ids``."""
        if logprobs_dims is None:
            return None

        num_logprobs, max_per_req_token_ids = logprobs_dims
        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = tilelang_inputs.flatten_sampled(
            sampled, num_sampled, cu_num_logits[: num_reqs + 1], num_logits
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logprob_token_ids_state=self.sampler.logprob_token_ids_state,
            expanded_idx_mapping=expanded_idx_mapping,
            max_per_req_token_ids=max_per_req_token_ids,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _watermarking_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        if self.watermark_key is None:
            return {}
        assert isinstance(self.sampler, GPUWatermarkSampler)
        return {
            "contexts": self.sampler._get_contexts(
                expanded_idx_mapping, expanded_local_pos, draft_sampled
            ),
            "watermarking": self.sampler.watermarking.gpu,
            "watermark_key": self.watermark_key,
        }

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        dead_row_cut: torch.Tensor | None = None,
        sink: _SinkChunk | None = None,
        kernel_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """With ``dead_row_cut`` ([num_reqs] int32), each request commits at most its
        cut plus one: block verification ends its block at the cut, the other
        methods cap the count (the insertion of the resampled token stores it so).
        ``kernel_steps``: the drafts every request of the step verifies at most,
        the width block verification's kernels run at."""
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        cut = None
        if sink is not None:
            assert self.sink is not None
            cut = self.sink.cut(
                processed_logits,
                sink.flags,
                draft_sampled,
                cu_num_logits,
                idx_mapping.shape[0],
            )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            dead_row_cut=dead_row_cut,
            kernel_steps=kernel_steps,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        if cut is not None:
            assert self.sink is not None and sink is not None
            # From the scores the last tokens were drawn from: the cut stays
            # applied until they are taken.
            sink.alternates = self.sink.alternates_of(
                processed_logits,
                cu_num_logits,
                num_sampled,
                sampled,
                draft_sampled,
                idx_mapping,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                pos,
                mask_rejected_draft=True,
                # A block that ends at the cut drew its last token as the bonus.
                dead_cut=dead_row_cut if self.use_block_verification else None,
            )
            cut.restore()
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        logprobs_dims: tuple[int, int] | None,
        dead_row_cut: torch.Tensor | None = None,
        sink_flags: torch.Tensor | None = None,
        sink_alternates: list[SinkAlternates] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        """With ``sink_flags`` (sink rejection), cuts the verification at
        flagged drafter rows and appends the alternates of each request's last
        draw to ``sink_alternates``. With ``dead_row_cut``, each request commits at
        most its cut plus one (``_verify``)."""
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs
        # Copy drafts: a step where no request was scheduled more than the
        # drafter's block (no copy) verifies at that width, as without them.
        kernel_steps = None
        drafts = input_batch.num_draft_tokens_per_req
        if self.num_draft_steps < self.num_speculative_steps and (
            drafts is None
            or min(int(drafts.max(initial=0)), input_batch.num_draft_tokens)
            <= self.num_draft_steps
        ):
            kernel_steps = self.num_draft_steps

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []
        alternates_chunks: list[SinkAlternates] = (
            sink_alternates if sink_alternates is not None else []
        )

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            if lo == 0:
                # The first chunk's offsets are the batch's own: a view, no launch.
                # (Adaptive verification's compacted offsets exist on the device
                # only, and its batch is always one chunk.)
                chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1]
            else:
                # A later chunk (never adaptive, so the host's offsets are the
                # device's): its offsets uploaded from the host, no kernel.
                chunk_cu_num_logits = async_tensor_h2d(
                    chunk_cu_num_logits_np, device=input_batch.cu_num_logits.device
                )
            # draft_logits uses persistent request-state indices and stays global.
            sink = None if sink_flags is None else _SinkChunk(sink_flags[lo:hi])
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                None if dead_row_cut is None else dead_row_cut[start:end],
                sink,
                kernel_steps,
            )
            if sink is not None:
                alternates = sink.alternates
                assert self.sink is not None and alternates is not None
                if logprobs_dims is not None:
                    alternates_chunks.append(
                        self.sink.alternate_logprobs_of(
                            processed_logits if use_processed_logits else logits[lo:hi],
                            chunk_cu_num_logits,
                            num_sampled,
                            alternates,
                            logits_mode=self.sampler.logprobs_mode
                            in ("raw_logits", "processed_logits"),
                        )
                    )
                else:
                    alternates_chunks.append(SinkAlternates(alternates))
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                input_batch.expanded_idx_mapping[lo:hi],
                logprobs_dims,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors
        # Several chunks (more than get_max_chunk_logits rows; never at DS4.1's
        # decode shapes): torch concatenates their results.

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        dead_row_cut: torch.Tensor | None = None,
        sink_flags: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """``sink_flags``: the sink detector's flag per logits row, with sink
        rejection on (vllm.v1.worker.gpu.sample.sink): flagged drafter rows cut
        the verification, and the output carries each request's alternates."""
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled, pos = logits_rows(input_batch)

        # As the sampler's: requests with logprob_token_ids get those ids'
        # logprobs for every verified token too.
        logprobs_dims = self.sampler.get_logprobs_dims(input_batch.idx_mapping_np)
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sink_rejection = sink_flags is not None and self.sink is not None
        alternates_chunks: list[SinkAlternates] = []
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            logprobs_dims,
            # Block verification ends each block at the cut; logprobs (and the
            # sink alternates) take their token counts inside the chunks.
            dead_row_cut,
            sink_flags if sink_rejection else None,
            alternates_chunks,
        )
        alternates = None
        if len(alternates_chunks) == 1:
            alternates = alternates_chunks[0]
        elif alternates_chunks:
            first = alternates_chunks[0]
            alternates = SinkAlternates(
                torch.cat([a.tokens for a in alternates_chunks]),
                torch.cat([a.logprobs for a in alternates_chunks])  # type: ignore[misc]
                if first.logprobs is not None
                else None,
                torch.cat([a.ranks for a in alternates_chunks])  # type: ignore[misc]
                if first.ranks is not None
                else None,
            )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
            dead_row_cut=dead_row_cut,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sink_alternates=alternates,
        )
