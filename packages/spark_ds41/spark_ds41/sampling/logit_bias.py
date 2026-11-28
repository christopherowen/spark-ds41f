# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from spark_ds41.sampling import tilelang_bias

MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128


class LogitBiasState:
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs

        # Allowed token IDs.
        self.num_allowed_token_ids = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.allowed_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        # Logit bias.
        self.num_logit_bias = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.logit_bias_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.int32,
            device=device,
        )
        self.logit_bias = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.float32,
            device=device,
        )
        # Min tokens.
        self.min_lens = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.restore_when_all_masked = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.stop_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

        # Using any of the above.
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        # Using any logit bias.
        use_logit_bias = False

        # Allowed token IDs.
        allowed_token_ids = sampling_params.allowed_token_ids
        if allowed_token_ids:
            num_allowed_token_ids = len(allowed_token_ids)
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:
                raise ValueError(
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."
                )
            self.num_allowed_token_ids.np[req_idx] = num_allowed_token_ids
            self.allowed_token_ids.stage_write(req_idx, 0, allowed_token_ids)
            use_logit_bias = True
        else:
            self.num_allowed_token_ids.np[req_idx] = 0

        # Logit bias.
        logit_bias = sampling_params.logit_bias
        if logit_bias:
            num_logit_bias = len(logit_bias)
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:
                raise ValueError(
                    f"Too many logit bias tokens: {num_logit_bias}. "
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."
                )
            self.num_logit_bias.np[req_idx] = num_logit_bias
            self.logit_bias_token_ids.stage_write(req_idx, 0, logit_bias.keys())
            self.logit_bias.stage_write(req_idx, 0, logit_bias.values())
            use_logit_bias = True
        else:
            self.num_logit_bias.np[req_idx] = 0

        # Min tokens.
        min_tokens = sampling_params.min_tokens
        min_len = prompt_len + min_tokens
        self.min_lens.np[req_idx] = min_len
        stop_token_ids = sampling_params.all_stop_token_ids
        if min_tokens > 0 and stop_token_ids:
            num_stop_token_ids = len(stop_token_ids)
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:
                raise ValueError(
                    f"Too many stop tokens: {num_stop_token_ids}. "
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."
                )
            self.num_stop_token_ids.np[req_idx] = num_stop_token_ids
            self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)
            self.restore_when_all_masked.np[req_idx] = int(
                sampling_params.structured_outputs is not None
            )
            use_logit_bias = True
        else:
            self.num_stop_token_ids.np[req_idx] = 0
            self.restore_when_all_masked.np[req_idx] = 0

        self.use_logit_bias[req_idx] = use_logit_bias

    def apply_staged_writes(self) -> None:
        self.num_allowed_token_ids.copy_to_uva()
        self.allowed_token_ids.apply_write()

        self.num_logit_bias.copy_to_uva()
        self.logit_bias_token_ids.apply_write()
        self.logit_bias.apply_write()

        self.min_lens.copy_to_uva()
        self.num_stop_token_ids.copy_to_uva()
        self.restore_when_all_masked.copy_to_uva()
        self.stop_token_ids.apply_write()

    def register_warmups(
        self, logits_strides: tuple[int, ...], vocab_size: int
    ) -> None:
        """Register the staged writes of the per-request lists and declare the
        bias kernel for processed FP32 logits with each row stride in
        ``logits_strides``, with and without the all-masked stop-token restore."""
        for tensor in (
            self.allowed_token_ids,
            self.logit_bias_token_ids,
            self.logit_bias,
            self.stop_token_ids,
        ):
            tensor.register_warmup()
        tilelang_bias.register_bias(
            vocab_size=vocab_size,
            logits_strides=logits_strides,
            allowed_stride=self.allowed_token_ids.gpu.stride(0),
            bias_ids_stride=self.logit_bias_token_ids.gpu.stride(0),
            bias_stride=self.logit_bias.gpu.stride(0),
            stop_stride=self.stop_token_ids.gpu.stride(0),
            width=next_power_of_2(
                max(
                    self.allowed_token_ids.gpu.shape[-1],
                    self.logit_bias_token_ids.gpu.shape[-1],
                    self.stop_token_ids.gpu.shape[-1],
                )
            ),
        )

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> None:
        if not np.any(self.use_logit_bias[idx_mapping_np]):
            # No request uses logit bias. Skip the kernel launch.
            return

        enable_stop_token_restore = bool(
            np.any(self.restore_when_all_masked.np[idx_mapping_np])
        )

        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            self.num_allowed_token_ids.gpu,
            self.allowed_token_ids.gpu,
            self.num_logit_bias.gpu,
            self.logit_bias_token_ids.gpu,
            self.logit_bias.gpu,
            self.min_lens.gpu,
            self.num_stop_token_ids.gpu,
            self.restore_when_all_masked.gpu,
            self.stop_token_ids.gpu,
            enable_stop_token_restore,
        )


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    restore_when_all_masked: torch.Tensor,
    stop_token_ids: torch.Tensor,
    check_all_masked_rows: bool = False,
) -> None:
    tilelang_bias.apply_logit_bias(
        logits,
        expanded_idx_mapping,
        pos,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        check_all_masked_rows,
    )
