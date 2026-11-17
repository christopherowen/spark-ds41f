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

import os
from collections.abc import Sequence
from typing import Any

import torch

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
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


def _use_fused_markov_greedy_step(tp_size: int, step: int, num_reqs: int) -> bool:
    """Whether draft position ``step`` fuses the Markov bias GEMM and shard
    argmax into one kernel or projects ``markov_bias_local`` and reduces with
    ``greedy_markov_shard_pair``.

    On TP == 1 the full 129,280-row Markov W2 exceeds GB10's L2 cache, so the
    fused projection wins at every step and batch size. On multi-rank TP (TP3 /
    TP4), each rank's 12-bit packed W2 shard (16.55 MB / 12.40 MB) stays L2-hot
    after position 0; unfused GEMM + ``greedy_markov_shard_pair`` is faster on
    L2-hot steps (``step >= 1``) and at ``num_reqs <= 2`` on position 0.
    """
    return tp_size <= 1 or (step == 0 and num_reqs >= 3)
from vllm.v1.worker.gpu.spec_decode.dspark.turn_learning import (
    DSparkTurnLearner,
    turn_learning_capacity,
    turn_learning_dump_dir,
    turn_learning_mode,
)
from vllm.v1.worker.gpu.spec_decode.dspark.turn_learning_shadow import (
    DSparkShadowLearner,
    turn_learning_lr,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


def _cpu_list(text: str) -> set[int]:
    """A CPU list such as 0-4,10-14."""
    cpus: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def _tasks() -> dict[str, str]:
    """This process's threads: id to name."""
    out: dict[str, str] = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as handle:
                out[tid] = handle.read().strip()
        except OSError:
            continue
    return out


def _cpus_allowed(tid: str) -> str:
    try:
        with open(f"/proc/self/task/{tid}/status") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def _needs_logits_for_acceptance(speculator: "DSparkSpeculator") -> bool:
    """Whether ``acceptance_estimator`` reads full vocabulary draft logits.

    When ``use_confidence_head`` is true, the estimator calibrates the scalar
    confidence head output instead of reducing draft logits, so local and
    vocabulary-parallel greedy draft reductions remain eligible.
    """
    return (
        getattr(speculator, "acceptance_estimator", None) is not None
        and not getattr(speculator, "use_confidence_head", False)
    )


def _resolve_active_spec_steps(
    speculator: "DSparkSpeculator", num_speculative_tokens: int | None
) -> int:
    """Clamp dynamic speculative token count to ``[1, num_speculative_steps]``."""
    n_spec = speculator.num_speculative_steps
    if num_speculative_tokens is None:
        return n_spec
    return max(1, min(int(num_speculative_tokens), n_spec))


def _select_sample_hidden(
    speculator: "DSparkSpeculator",
    head_hidden: torch.Tensor,
    num_reqs: int,
    active_steps: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return per-(req, active_step) hidden states before the base draft lm_head."""
    n_spec = speculator.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_from_anchor = getattr(speculator, "sample_from_anchor", False)
    if active_steps == n_spec:
        sample_hidden = (
            head_hidden[:num_sample]
            if sample_from_anchor
            else tilelang_inputs.gather_hidden_rows(
                head_hidden, speculator.sample_indices[:num_sample]
            )
        )
        return sample_hidden, None
    if sample_from_anchor:
        conf_sample_hidden = head_hidden[:num_sample]
        sample_hidden = (
            conf_sample_hidden.view(num_reqs, n_spec, -1)[:, :active_steps]
            .reshape(num_reqs * active_steps, -1)
            .contiguous()
        )
        return sample_hidden, conf_sample_hidden
    active_indices = (
        speculator.sample_indices[:num_sample]
        .view(num_reqs, n_spec)[:, :active_steps]
        .reshape(-1)
        .contiguous()
    )
    return tilelang_inputs.gather_hidden_rows(head_hidden, active_indices), None


def _compute_confidence_probs(
    speculator: "DSparkSpeculator",
    head_hidden: torch.Tensor,
    sample_hidden: torch.Tensor,
    conf_sample_hidden: torch.Tensor | None,
    confidence_embeds: torch.Tensor | None,
    confidence_markov_embeds: list[torch.Tensor],
    num_reqs: int,
    active_steps: int,
) -> None:
    """Evaluate and calibrate DSpark confidence probabilities over active draft steps."""
    n_spec = speculator.num_speculative_steps
    num_sample = num_reqs * n_spec
    truncated = active_steps < n_spec
    if confidence_embeds is not None:
        conf_embeds = confidence_embeds.flatten(0, 1)
    else:
        stacked = torch.stack(confidence_markov_embeds, dim=1)
        if not truncated:
            conf_embeds = stacked.flatten(0, 1)
        else:
            padded = torch.zeros(
                (num_reqs, n_spec, stacked.shape[-1]),
                dtype=stacked.dtype,
                device=stacked.device,
            )
            padded[:, :active_steps].copy_(stacked)
            conf_embeds = padded.flatten(0, 1)

    if not truncated:
        conf_hidden = sample_hidden
        conf_idx_mapping = speculator.sample_idx_mapping[:num_sample]
    else:
        conf_hidden = (
            conf_sample_hidden
            if conf_sample_hidden is not None
            else tilelang_inputs.gather_hidden_rows(
                head_hidden, speculator.sample_indices[:num_sample]
            )
        )
        # Mask tail slots to -1 so calibration neither writes tail features nor predictions.
        conf_idx_mapping = speculator.sample_idx_mapping[:num_sample].clone()
        conf_idx_mapping.view(num_reqs, n_spec)[:, active_steps:].fill_(-1)

    conf_out = speculator.draft_token_confidence_probs[:num_reqs].view(-1)
    estimator = getattr(speculator, "acceptance_estimator", None)
    if estimator is not None and hasattr(
        speculator.model, "compute_and_calibrate_confidence"
    ):
        speculator.model.compute_and_calibrate_confidence(
            conf_hidden,
            conf_embeds,
            conf_out,
            estimator,
            conf_idx_mapping,
        )
    else:
        speculator.model.compute_confidence(
            conf_hidden,
            conf_embeds,
            out=conf_out,
        )
        if estimator is not None:
            estimator.calibrate_confidence_probs(
                speculator.draft_token_confidence_probs[:num_reqs],
                conf_idx_mapping,
            )
    if truncated:
        speculator.draft_token_confidence_probs[:num_reqs, active_steps:].zero_()


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
        self._top2_branch_margin: float = float(
            getattr(
                self.draft_model_config.hf_config,
                "dspark_top2_branch_margin",
                0.0,
            )
        )
        self._top2_branch_weight: float = float(
            getattr(
                self.draft_model_config.hf_config,
                "dspark_top2_branch_weight",
                1.0,
            )
        )
        self._enable_top2_branch: bool = (
            bool(
                getattr(
                    self.draft_model_config.hf_config,
                    "dspark_top2_branch",
                    False,
                )
            )
            or self._top2_branch_margin > 0.0
        )
        draft_vocab = max(
            self.draft_model_config.hf_config.vocab_size,
            getattr(self.draft_model_config.hf_config, "draft_vocab_size", None) or 0,
        )
        shape = scratch_shape(
            self.max_num_reqs, draft_vocab, top2=self._enable_top2_branch
        )
        self._greedy_partial_values = torch.empty(
            shape, dtype=torch.float32, device=device
        )
        self._greedy_partial_indices = torch.empty(
            shape, dtype=torch.int32, device=device
        )
        # Per-request (max, global index) of this rank's vocabulary shard,
        # for vocabulary-parallel greedy drafts.
        # (value, index) and two slack columns (used for shard runner-up
        # (second_value, second_index) when top-2 branch tracking is enabled):
        # 16-byte rows, which the one-shot all-gather writes in place.
        self._greedy_shard_pairs = torch.empty(
            (self.max_num_reqs, 4), dtype=torch.float32, device=device
        )
        self.draft_runner_up_tokens = torch.zeros_like(self.draft_tokens)
        self.draft_top2_margins = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
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
        self._markov_conf_weight = torch.zeros(
            rank, dtype=torch.float32, device=device
        )
        logger.info_once("DSpark pre-drafting Markov head truncation enabled")
        # VLLM_DSPARK_TURN_LEARNING: end-of-turn replay of each request's drafts
        # (turn_learning.py, observe) or a shadow copy of Markov W1 trained on
        # each finished turn (turn_learning_shadow.py); None when off.
        self.turn_learner: DSparkTurnLearner | DSparkShadowLearner | None = None

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
        # Keep the online acceptance estimator active when a trained confidence
        # head is present so it calibrates the head's per-position survival and
        # slope from observed target verification outcomes.
        self.use_acceptance_estimator = self.enable_adaptive_verification
        head = getattr(model.model, "confidence_head", None)
        proj = getattr(head, "proj", None)
        weight = getattr(proj, "weight", None)
        rank = self._markov_step_embed.shape[1]
        hidden = self.hidden_states.shape[1]
        if (
            weight is not None
            and weight.dtype == torch.float32
            and weight.numel() == hidden + rank
            and rank > 0
        ):
            self._markov_conf_weight = weight.view(-1)[hidden:].contiguous()
        return model

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        if self.use_confidence_head and self.acceptance_estimator is not None:
            self.acceptance_estimator.reset_for_confidence_head()
        self._enable_turn_learning()

    def _enable_turn_learning(self) -> None:
        """Attach ``VLLM_DSPARK_TURN_LEARNING``'s learner, before the draft
        kernels are declared and the draft graphs captured. It replays the
        vocabulary-parallel greedy draft step and refuses any other."""
        mode = turn_learning_mode()
        if mode == "off":
            if os.environ.get("VLLM_DSPARK_TURN_LEARNING_COMM_ONLY", "") == "1":
                self._turn_learning_comm_only()
            return
        local_head = getattr(self.model, "supports_local_draft_argmax", lambda: False)()
        table = self._markov_table()
        tp_size = get_tensor_model_parallel_world_size()
        problems = [
            name
            for name, bad in (
                ("sample_from_anchor", not self.sample_from_anchor),
                ("vocabulary-parallel greedy drafts", not self._vocab_parallel_greedy(local_head)),
                ("a replicated BF16 Markov W1", table is None),
                ("tensor parallelism", tp_size <= 1),
                ("no top-k drafts", self._draft_topk is not None),
                ("no top-2 branch", self._enable_top2_branch),
                ("the unfused Markov bias", not hasattr(self.model, "markov_bias_local")),
            )
            if bad
        ]
        if problems:
            raise RuntimeError(
                "VLLM_DSPARK_TURN_LEARNING replays the vocabulary-parallel greedy "
                "draft step only; this one lacks " + ", ".join(problems)
            )
        assert table is not None
        if mode == "shadow":
            # The end of turn runs on the learner's stream while the step's
            # collectives run on the step's: its own NCCL communicator over the
            # tensor-parallel group, never the step's.
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.parallel_state import get_tp_group

            comm = PyNcclCommunicator(get_tp_group().cpu_group, device=self.device)
            if comm.disabled:
                raise RuntimeError(
                    "shadow turn learning needs its own NCCL communicator over the "
                    "tensor-parallel group; NCCL is unavailable"
                )
            self._turn_learning_comm = comm
            self.turn_learner = DSparkShadowLearner(
                model=self.model,
                table=table,
                markov_w2=self.model.model.markov_head.markov_w2,
                draft_tokens=self.draft_tokens,
                input_ids=self.input_buffers.input_ids,
                anchor_idx=self._anchor_idx,
                sample_idx_mapping=self.sample_idx_mapping,
                sample_pos=self.sample_pos,
                gather_into=lambda out, x: comm.all_gather(
                    out, x, stream=torch.cuda.current_stream()
                ),
                reduce_in_place=lambda x: comm.all_reduce(
                    x, out_tensor=x, stream=torch.cuda.current_stream()
                ),
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                tp_size=tp_size,
                tp_rank=get_tensor_model_parallel_rank(),
                device=self.device,
                capacity=turn_learning_capacity(),
                lr=turn_learning_lr(),
            )
            return
        self.turn_learner = DSparkTurnLearner(
            model=self.model,
            table=table,
            draft_tokens=self.draft_tokens,
            input_ids=self.input_buffers.input_ids,
            anchor_idx=self._anchor_idx,
            sample_idx_mapping=self.sample_idx_mapping,
            sample_pos=self.sample_pos,
            partial_values=self._greedy_partial_values,
            partial_indices=self._greedy_partial_indices,
            fused_step=_use_fused_markov_greedy_step,
            all_gather=lambda pairs: tensor_model_parallel_all_gather(pairs, dim=-1),
            max_num_reqs=self.max_num_reqs,
            num_speculative_steps=self.num_speculative_steps,
            hidden_size=self.hidden_states.shape[1],
            tp_size=tp_size,
            tp_rank=get_tensor_model_parallel_rank(),
            device=self.device,
            capacity=turn_learning_capacity(),
            dump_dir=turn_learning_dump_dir(),
        )

    def _turn_learning_comm_only(self) -> None:
        """Attribution arm (overlay-commthreads/, VLLM_DSPARK_TURN_LEARNING_COMM_ONLY=1
        with the learner off): the shadow learner's own NCCL communicator over the
        tensor-parallel group, made as shadow makes it and exercised at a chunk's sizes
        (the all-gather of 256 entries' summaries, the all-reduce of their FP32 gradient
        rows) on a side stream at load. Then, by
        VLLM_DSPARK_TURN_LEARNING_COMM_CPUS (a list such as 0-4,10-14), it is made
        and exercised from a thread confined to those cores, so the threads NCCL
        starts meanwhile inherit them; by VLLM_DSPARK_TURN_LEARNING_COMM_DESTROY=1, it
        is destroyed (ncclCommAbort, vLLM's destroy) once exercised. The threads it
        started, their cores, and what remains after a destroy are logged."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_tp_group

        cpus = _cpu_list(os.environ.get("VLLM_DSPARK_TURN_LEARNING_COMM_CPUS", ""))
        destroy = os.environ.get("VLLM_DSPARK_TURN_LEARNING_COMM_DESTROY", "") == "1"
        before = _tasks()
        saved = os.sched_getaffinity(0) if cpus else None
        if cpus:
            cpus &= saved
            if not cpus:
                raise RuntimeError(
                    "VLLM_DSPARK_TURN_LEARNING_COMM_CPUS names no core this process may use"
                )
            os.sched_setaffinity(0, cpus)
        try:
            comm = PyNcclCommunicator(get_tp_group().cpu_group, device=self.device)
            if comm.disabled:
                raise RuntimeError("VLLM_DSPARK_TURN_LEARNING_COMM_ONLY: NCCL is unavailable")
            self._turn_learning_comm = comm
            tp_size = get_tensor_model_parallel_world_size()
            rows = 256 * self.num_speculative_steps
            summary = torch.zeros(rows * 10, dtype=torch.float32, device=self.device)
            gathered = torch.empty(tp_size * rows * 10, dtype=torch.float32, device=self.device)
            grad = torch.zeros(rows * 256, dtype=torch.float32, device=self.device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(16):
                    comm.all_gather(gathered, summary, stream=side)
                    comm.all_reduce(grad, out_tensor=grad, stream=side)
            side.synchronize()
        finally:
            if saved is not None:
                os.sched_setaffinity(0, saved)
        made = {tid: name for tid, name in _tasks().items() if tid not in before}
        logger.info(
            "DSpark turn learning off; the shadow learner's NCCL communicator made "
            "and exercised at a chunk's sizes%s (attribution arm); threads started "
            "meanwhile: %s",
            f" from cores {sorted(cpus)}" if cpus else "",
            ", ".join(f"{name}[{_cpus_allowed(tid)}]" for tid, name in sorted(made.items()))
            or "none",
        )
        if destroy:
            # ncclCommAbort is local and immediate: a rank done with the exercise
            # must not abort while a peer's last transfer is still in flight
            # (without this barrier, ranks 1 and 3 hung in their last all-reduce on
            # 2026-11-17 and the boot timed out).
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            comm.destroy()
            self._turn_learning_comm = None
            left = {tid: name for tid, name in _tasks().items() if tid in made}
            logger.info(
                "DSpark turn learning off; the communicator destroyed (attribution "
                "arm); of its threads still running: %s",
                ", ".join(f"{name}" for name in left.values()) or "none",
            )

    def propose(self, input_batch: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        draft_tokens = super().propose(input_batch, *args, **kwargs)
        if (
            self.turn_learner is not None
            and not kwargs.get("dummy_run", False)
            and not kwargs.get("is_profile", False)
        ):
            self.turn_learner.note_drafts(
                input_batch.idx_mapping_np[: input_batch.num_reqs]
            )
        return draft_tokens

    @property
    def tp_invariant_confidence(self) -> bool:
        """Whether every TP rank computes bitwise-identical draft confidences
        when TP reductions are rank-ordered."""
        if not self.use_confidence_head:
            return False
        head = getattr(getattr(self.model, "model", None), "confidence_head", None)
        return bool(getattr(head, "tp_invariant", False))

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if not self.use_confidence_head:
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
            and not _needs_logits_for_acceptance(self)
        )

    def register_warmups(
        self, block_table_layouts: Sequence[DraftBlockTableLayout]
    ) -> None:
        """Also register the greedy reduction of each draft position: the
        vocabulary-parallel one's selection over the ranks' shard pairs (draft
        position ``i`` is column ``i`` of the draft tokens; the Markov head
        declares the fused projection that reduces each shard, and a head
        without one fails here); or the full-vocabulary one, when no logits
        cache, vocabulary map or estimator needs the logits."""
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
                and not _needs_logits_for_acceptance(self)
            ):
                register_sample_greedy_markov_warmup(
                    dtype=processor.head_dtype, output_dtype=self.draft_tokens.dtype
                )
            return
        if not self.model.fuses_markov_greedy:
            raise RuntimeError(
                "Vocabulary-parallel greedy drafts reduce each shard in the Markov "
                "projection's epilogue, which only a TileLang packed or BF16 Markov "
                "head without a logit scale or soft cap declares"
            )
        n_spec = self.num_speculative_steps
        use_top2 = self._enable_top2_branch
        tp_size = get_tensor_model_parallel_world_size()
        has_local_layout = hasattr(self.model, "draft_logits_local_layout")
        if tp_size > 1 and has_local_layout:
            vocab_start, shard_vocab = self.model.draft_vocab_shard
            dtype, head_width, markov_width = self.model.draft_logits_local_layout
        for i in range(n_spec):
            if tp_size > 1 and has_local_layout:
                register_greedy_markov_shard_pair_warmup(
                    base_dtype=dtype,
                    base_stride=n_spec * head_width,
                    base_aligned=i * head_width * dtype.itemsize % 16 == 0,
                    bias_dtype=dtype,
                    bias_stride=markov_width,
                    vocab=shard_vocab,
                    vocab_start=vocab_start,
                    top2=use_top2,
                )
            register_select_first_argmax_warmup(
                ranks=get_tensor_model_parallel_world_size(),
                output_dtype=self.draft_tokens.dtype,
                output_stride=self.draft_tokens.stride(0),
                output_aligned=i * self.draft_tokens.element_size() % 16 == 0,
                top2=use_top2,
            )
        if self.turn_learner is not None:
            tilelang_inputs.declare(self.turn_learner.calls)
        if not getattr(self, "sample_from_anchor", False):
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
                    top2_branch=use_top2,
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

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        num_speculative_tokens: int | None = None,
    ) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(
                num_reqs,
                head_hidden,
                num_speculative_tokens=num_speculative_tokens,
            )
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        active_steps = _resolve_active_spec_steps(self, num_speculative_tokens)
        num_sample = num_reqs * n_spec
        sample_hidden, conf_sample_hidden = _select_sample_hidden(
            self, head_hidden, num_reqs, active_steps
        )
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        local_head = getattr(self.model, "supports_local_draft_argmax", lambda: False)()
        vocab_parallel = self._vocab_parallel_greedy(local_head)
        can_unfuse_markov = vocab_parallel and hasattr(self.model, "markov_bias_local")
        tp_size = get_tensor_model_parallel_world_size() if can_unfuse_markov else 1
        if local_head:
            base_logits = self.model.compute_local_draft_logits(sample_hidden)
        elif vocab_parallel:
            if can_unfuse_markov:
                vocab_start, shard_vocab = self.model.draft_vocab_shard
            base_logits = self.model.compute_draft_logits_local(sample_hidden)
        else:
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, active_steps, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        if active_steps < n_spec:
            idx_map = idx_map[:, :active_steps]
            sample_pos = sample_pos[:, :active_steps]
        confidence_markov_embeds: list[torch.Tensor] = []

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
            confidence_embeds = None
            prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        shadow = (
            self.turn_learner
            if isinstance(self.turn_learner, DSparkShadowLearner) and vocab_parallel
            else None
        )
        shadow_fused: tuple[int, int] | None = None
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
                if self.draft_logits is None and not _needs_logits_for_acceptance(self):
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
                use_top2 = self._enable_top2_branch
                pairs = self._greedy_shard_pairs[:num_reqs]
                if not can_unfuse_markov or _use_fused_markov_greedy_step(
                    tp_size, i, num_reqs
                ):
                    # The Markov projection adds the base logits and reduces
                    # this rank's shard: no bias tensor.
                    if shadow is not None and i == 0:
                        # The same kernels, the tiles' maxima kept for the learner.
                        shadow_fused = shadow.fused_pairs(
                            markov_embed, base_logits[:, i], pairs
                        )
                    else:
                        self.model.markov_greedy_pairs_local(
                            markov_embed, base_logits[:, i], pairs, top2=use_top2
                        )
                else:
                    bias = self.model.markov_bias_local(markov_embed)
                    partial_values, partial_indices = (
                        shadow.partials(i)
                        if shadow is not None
                        else (self._greedy_partial_values, self._greedy_partial_indices)
                    )
                    greedy_markov_shard_pair(
                        base_logits[:, i, :shard_vocab],
                        bias[:, :shard_vocab],
                        vocab_start,
                        pairs,
                        partial_values,
                        partial_indices,
                        top2=use_top2,
                    )
                if shadow is not None and i + 1 == active_steps:
                    # Every position's maxima are written: keep them on the side
                    # stream while this position's pairs are exchanged.
                    shadow.capture(num_reqs, active_steps, shadow_fused)
                draft_sampled_i = self.draft_tokens[:num_reqs, i]
                has_embed = table is not None and i + 1 < active_steps
                select_first_argmax(
                    tensor_model_parallel_all_gather(pairs, dim=-1),
                    draft_sampled_i,
                    pair_width=4,
                    embed=(
                        (table, step_embed, confidence_embeds[:, i + 1])
                        if has_embed
                        else None
                    ),
                    top2_branch=(
                        (
                            self._markov_conf_weight if has_embed else None,
                            self.draft_runner_up_tokens[:num_reqs, i],
                            self.draft_top2_margins[:num_reqs, i],
                            self._top2_branch_margin,
                            self._top2_branch_weight,
                        )
                        if use_top2
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
                and not _needs_logits_for_acceptance(self)
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

        if self.use_confidence_head:
            _compute_confidence_probs(
                self,
                head_hidden,
                sample_hidden,
                conf_sample_hidden,
                confidence_embeds if table is not None else None,
                confidence_markov_embeds,
                num_reqs,
                active_steps,
            )
        if isinstance(self.turn_learner, DSparkTurnLearner):
            # Keep the step's head rows, anchor position and drafts in each
            # request's ring (inside the draft graph).
            self.turn_learner.capture(num_reqs, head_hidden, active_steps)
        elif shadow is not None:
            shadow.join()

    def _sample_sequential_topk(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        num_speculative_tokens: int | None = None,
    ) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        active_steps = _resolve_active_spec_steps(self, num_speculative_tokens)
        num_sample = num_reqs * n_spec
        sample_hidden, conf_sample_hidden = _select_sample_hidden(
            self, head_hidden, num_reqs, active_steps
        )
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, active_steps, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        if active_steps < n_spec:
            idx_map = idx_map[:, :active_steps]
            sample_pos = sample_pos[:, :active_steps]
        confidence_markov_embeds: list[torch.Tensor] = []
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

        if self.use_confidence_head:
            _compute_confidence_probs(
                self,
                head_hidden,
                sample_hidden,
                conf_sample_hidden,
                None,
                confidence_markov_embeds,
                num_reqs,
                active_steps,
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        num_speculative_tokens: int | None = None,
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
        self._sample_sequential(
            num_reqs,
            head_hidden,
            num_speculative_tokens=num_speculative_tokens,
        )
