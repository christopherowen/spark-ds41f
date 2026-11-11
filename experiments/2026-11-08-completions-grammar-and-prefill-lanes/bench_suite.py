#!/usr/bin/env python3
import functools
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.reasoning import ReasoningParserManager
DeepSeekV41ParserReasoningAdapter = ReasoningParserManager.get_reasoning_parser('deepseek_v41')
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
import xgrammar as xgr

DS41_SNAPSHOT = (
    "/mnt/ds41-hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
    "snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
)
JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
)


def _row_pattern(bitmask: torch.Tensor) -> str:
    return "".join("U" if bool((row == -1).all()) else "C" for row in bitmask)


def bench_structured_output(tmp_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(DS41_SNAPSHOT, trust_remote_code=True)
    so_model_dir = tmp_dir / "so_model"
    so_model_dir.mkdir(parents=True, exist_ok=True)
    (so_model_dir / "config.json").write_text(json.dumps({
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "ffn_dim": 3072,
        "max_position_embeddings": 8192,
        "vocab_size": len(tokenizer),
        "word_embed_proj_dim": 768,
        "torch_dtype": "float16",
        "do_layer_norm_before": True,
        "pad_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 2,
    }))
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=str(so_model_dir), tokenizer=DS41_SNAPSHOT, trust_remote_code=True),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar",
            enable_in_reasoning=False,
        ),
        speculative_config=SpeculativeConfig(
            model="[ngram]",
            num_speculative_tokens=5,
        ),
    )

    vllm_config.model_config.hf_config.vocab_size = len(tokenizer)
    vllm_config.model_config.get_vocab_size = lambda: len(tokenizer)
    # Token IDs in DeepSeek V4.1 tokenizer
    tok_open = tokenizer.encode("{", add_special_tokens=False)
    tok_quote = tokenizer.encode('"', add_special_tokens=False)
    tok_invalid = tokenizer.encode("thinking", add_special_tokens=False)
    tok_think_end = tokenizer.encode("</think>", add_special_tokens=False)
    assert len(tok_open) == 1 and len(tok_invalid) == 1 and len(tok_think_end) == 1
    open_id = tok_open[0]
    quote_id = tok_quote[0]
    invalid_id = tok_invalid[0]
    think_end_id = tok_think_end[0]
    prompt_ids = tokenizer.encode("Output JSON only:", add_special_tokens=False)

    def run_mode(patched: bool, is_chat: bool, with_think_end_in_draft: bool = False):
        manager = StructuredOutputManager(vllm_config)
        manager.tokenizer = tokenizer
        manager.reasoner_cls = DeepSeekV41ParserReasoningAdapter

        orig_get_cs = StructuredOutputManager._get_constraint_start
        if not patched:
            # Simulate pre-fix behavior where reasoning_parser_kwargs is not checked
            def unpatched_get_cs(self, request, spec_tokens, spec_tokens_committed=False):
                if self.enable_in_reasoning:
                    return 0
                sreq = request.structured_output_request
                if sreq.reasoning_ended:
                    return 0
                reasoner = self._get_reasoner(request)
                if reasoner is None:
                    return 0
                if sreq.reasoning_ended is None:
                    if reasoner.is_reasoning_end(request.prompt_token_ids or []):
                        return 0
                    sreq.reasoning_ended = False
                return orig_get_cs(self, request, spec_tokens, spec_tokens_committed)

            manager._get_constraint_start = functools.partial(unpatched_get_cs, manager)

        sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
        sp.structured_outputs._backend = "xgrammar"
        sp.update_from_generation_config({}, tokenizer.eos_token_id)

        req = Request(
            request_id="req-bench",
            prompt_token_ids=list(prompt_ids),
            sampling_params=sp,
            pooling_params=None,
            reasoning_ended=False if is_chat else None,
            reasoning_parser_kwargs={"chat_template_kwargs": {}} if is_chat else None,
        )
        manager.grammar_init(req)
        sreq = req.structured_output_request
        while not sreq._check_grammar_completion():
            time.sleep(0.005)

        if with_think_end_in_draft:
            drafts = [invalid_id, think_end_id, open_id, invalid_id, quote_id]
        else:
            drafts = [open_id, invalid_id, quote_id, open_id, open_id]

        cs = manager._get_constraint_start(req, drafts, spec_tokens_committed=False)
        trimmed = manager.trim_spec_tokens(req, drafts)
        bitmask = manager.grammar_bitmask(
            requests={req.request_id: req},
            structured_output_request_ids=[req.request_id],
            scheduled_spec_decode_tokens={req.request_id: trimmed},
        )
        pattern = _row_pattern(bitmask)

        # GPU bitmask application on row 0 (position 0 where open_id is valid and invalid_id is invalid)
        vocab_size = len(tokenizer)
        logits = torch.zeros((len(trimmed) + 1, vocab_size), dtype=torch.float32, device="cuda")
        logits[:, invalid_id] = 20.0
        logits[:, open_id] = 10.0
        bitmask_gpu = torch.from_numpy(bitmask).to(device="cuda", non_blocking=True)
        xgr.apply_token_bitmask_inplace(logits, bitmask_gpu)
        torch.cuda.synchronize()
        row0_top_token = int(torch.argmax(logits[0]).item())
        row0_invalid_masked = bool(logits[0, invalid_id].item() == float("-inf"))

        # Timing CPU manager step (trim + bitmask) over 500 iterations
        t0 = time.perf_counter()
        for _ in range(500):
            sreq.grammar.reset()
            if not is_chat:
                sreq.reasoning_ended = None
                sreq.reasoner = None
            else:
                sreq.reasoning_ended = False
            tr = manager.trim_spec_tokens(req, drafts)
            bm = manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids=[req.request_id],
                scheduled_spec_decode_tokens={req.request_id: tr},
            )
        cpu_us = (time.perf_counter() - t0) * 1e6 / 500.0

        # Timing GPU apply_grammar_bitmask over 500 iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            xgr.apply_token_bitmask_inplace(logits, bitmask_gpu)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(500):
            xgr.apply_token_bitmask_inplace(logits, bitmask_gpu)
        end_ev.record()
        torch.cuda.synchronize()
        gpu_us = start_ev.elapsed_time(end_ev) * 1000.0 / 500.0

        return {
            "constraint_start": cs,
            "drafts_len": len(drafts),
            "trimmed_len": len(trimmed),
            "bitmask_pattern": pattern,
            "reasoner_instantiated": sreq.reasoner is not None,
            "row0_invalid_masked": row0_invalid_masked,
            "row0_sampled_valid_open_brace": row0_top_token == open_id,
            "cpu_step_us": round(cpu_us, 2),
            "gpu_bitmask_us": round(gpu_us, 2),
        }

    return {
        "completions_unpatched": run_mode(patched=False, is_chat=False),
        "completions_patched": run_mode(patched=True, is_chat=False),
        "chat_thinking_active": run_mode(patched=True, is_chat=True, with_think_end_in_draft=False),
        "chat_think_end_mid_draft": run_mode(patched=True, is_chat=True, with_think_end_in_draft=True),
    }


def bench_scheduler(tmp_dir: Path) -> dict:
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    init_none_hash(sha256)
    block_size = 64
    block_hasher = get_request_block_hasher(block_size, sha256)

    opt_dir = tmp_dir / "opt_sched_model"
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "config.json").write_text(json.dumps({
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "ffn_dim": 3072,
        "max_position_embeddings": 524288,
        "vocab_size": 50272,
        "word_embed_proj_dim": 768,
        "torch_dtype": "float16",
        "do_layer_norm_before": True,
        "pad_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 2,
    }))

    BUDGET = 16384
    SEQS = 48
    ALIGNED_CHUNK = BUDGET - 2 * SEQS  # 16288 (promoted tp4-ring geometry: 48 seqs x 2 rows)

    def make_sched(
        *,
        align_prefill_chunks: bool,
        prefill_yield: str,
        max_parallel_prefills: int | str,
        prefill_long_threshold: int = 0,
        max_long_prefills: int = 1,
    ) -> Scheduler:
        model_config = ModelConfig(
            model=str(opt_dir),
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
            max_model_len=524288,
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=SEQS,
            max_num_batched_tokens=BUDGET,
            max_model_len=524288,
            long_prefill_token_threshold=ALIGNED_CHUNK,
            align_prefill_chunks=align_prefill_chunks,
            prefill_yield=prefill_yield,
            prefill_yield_limit_s=10.0,
            max_parallel_prefills=max_parallel_prefills,
            prefill_policy="round-robin",
            prefill_long_threshold=prefill_long_threshold,
            max_long_prefills=max_long_prefills,
            enable_chunked_prefill=True,
            async_scheduling=False,
            is_encoder_decoder=False,
            watermark=0.0,
        )
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        )
        spec_config = SpeculativeConfig(
            model="ngram",
            num_speculative_tokens=1,
            prompt_lookup_max=1,
            prompt_lookup_min=1,
        )
        vllm_config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(pipeline_parallel_size=1, data_parallel_size=1),
            speculative_config=spec_config,
        )
        kv_spec = FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        num_blocks = 40000
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_spec)],
        )
        cache_config.num_gpu_blocks = num_blocks
        register_all_kvcache_specs(vllm_config)
        return Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=block_size,
            log_stats=False,
            structured_output_manager=StructuredOutputManager(vllm_config),
        )

    def make_req(req_id: str, tokens: list[int], max_tokens: int = 32) -> Request:
        sp = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
        sp.update_from_generation_config({}, 2)
        return Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=sp,
            pooling_params=None,
            arrival_time=0.0,
            block_hasher=block_hasher,
        )

    def step_time_s(
        scheduled_map: dict[str, int],
        pre_step_computed: dict[str, int],
        req_map: dict[str, Request],
    ) -> float:
        total_tok = sum(scheduled_map.values())
        if total_tok == 0:
            return 0.0
        has_prefill = any(
            pre_step_computed[rid] < req_map[rid].num_prompt_tokens
            for rid in scheduled_map
        )
        if not has_prefill:
            return 0.0568
        max_ctx = max(
            pre_step_computed[rid] + tok
            for rid, tok in scheduled_map.items()
        )
        attn_ms = 0.25 * (total_tok / 1024.0) * (max_ctx / 8192.0)
        return (153.4 + 19.6 * (total_tok / 1024.0) + attn_ms) / 1000.0

    def run_scenario(
        arm_name: str,
        long_tokens: int,
        *,
        align_prefill_chunks: bool,
        prefill_yield: str,
        max_parallel_prefills: int | str,
        prefill_long_threshold: int = 0,
        max_long_prefills: int = 1,
    ) -> dict:
        sched = make_sched(
            align_prefill_chunks=align_prefill_chunks,
            prefill_yield=prefill_yield,
            max_parallel_prefills=max_parallel_prefills,
            prefill_long_threshold=prefill_long_threshold,
            max_long_prefills=max_long_prefills,
        )
        req_map: dict[str, Request] = {}

        # Populate prefix cache with a 64,000-token shared prefix
        shared_prefix = [100 + (i % 900) for i in range(64000)]
        warm_req = make_req("warm_prefix", shared_prefix, max_tokens=1)
        req_map["warm_prefix"] = warm_req
        sched.add_request(warm_req)
        while warm_req.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            out = sched.schedule()
            rids = list(out.num_scheduled_tokens)
            sampled = [[] if sched.requests[r].is_prefill_chunk else [10] for r in rids]
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=rids,
                    req_id_to_index={r: idx for idx, r in enumerate(rids)},
                    sampled_token_ids=sampled,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )

        # Start 8 active decode streams (128 tokens each) and ensure all 8 are in decode
        for d in range(8):
            dreq = make_req(f"decode_{d}", [2000 + d * 300 + i for i in range(128)], max_tokens=512)
            req_map[dreq.request_id] = dreq
            sched.add_request(dreq)
        while sched.waiting:
            out = sched.schedule()
            rids = list(out.num_scheduled_tokens)
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=rids,
                    req_id_to_index={r: idx for idx, r in enumerate(rids)},
                    sampled_token_ids=[[11] for _ in rids],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )

        # Submit long1 (long_tokens) and run its first prefill step (chunk 0)
        long1_toks = [10000 + (i % 5000) for i in range(long_tokens)]
        long1 = make_req("long1", long1_toks, max_tokens=16)
        req_map["long1"] = long1
        sched.add_request(long1)

        out0 = sched.schedule()
        rids0 = list(out0.num_scheduled_tokens)
        sampled0 = [[] if sched.requests[r].is_prefill_chunk else [12] for r in rids0]
        sched.update_from_output(
            out0,
            ModelRunnerOutput(
                req_ids=rids0,
                req_id_to_index={r: idx for idx, r in enumerate(rids0)},
                sampled_token_ids=sampled0,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

        # Now while long1 is mid-prefill (after chunk 1), arrive:
        # 1. Three additional long requests (long2, long3, long4) queued ahead of short requests
        # 2. A short cold interactive request (1,000 tokens)
        # 3. A short prefix-cached request (64,000 cached prefix + 1,000 uncached tail = 65,000 tokens)
        for k in (2, 3, 4):
            lk_toks = [10000 * k + (i % 5000) for i in range(long_tokens)]
            lk = make_req(f"long{k}", lk_toks, max_tokens=16)
            req_map[f"long{k}"] = lk
            sched.add_request(lk)

        short_cold = make_req("short_cold", [50000 + i for i in range(1000)], max_tokens=16)
        short_cached = make_req(
            "short_cached",
            shared_prefix + [60000 + i for i in range(1000)],
            max_tokens=16,
        )
        req_map["short_cold"] = short_cold
        req_map["short_cached"] = short_cached
        sched.add_request(short_cold)
        sched.add_request(short_cached)

        sim_clock_s = 0.0
        arrival_s = 0.0
        short_cold_admit_step = None
        short_cold_done_step = None
        short_cold_ttft_s = None
        short_cached_admit_step = None
        short_cached_done_step = None
        short_cached_ttft_s = None
        long1_done_step = None
        long1_ttft_s = None
        long1_chunks = [out0.num_scheduled_tokens.get("long1", 0)]

        t_cpu0 = time.perf_counter()
        step_idx = 0
        while (
            short_cold_done_step is None
            or short_cached_done_step is None
            or long1_done_step is None
        ) and step_idx < 400:
            step_idx += 1
            pre_step_computed = {rid: r.num_computed_tokens for rid, r in req_map.items()}
            out = sched.schedule()
            sched_map = dict(out.num_scheduled_tokens)
            dt = step_time_s(sched_map, pre_step_computed, req_map)
            sim_clock_s += dt

            if (
                "long1" in sched_map
                and pre_step_computed["long1"] < long1.num_prompt_tokens
            ):
                long1_chunks.append(sched_map["long1"])

            if "short_cold" in sched_map and short_cold_admit_step is None:
                short_cold_admit_step = step_idx
            if "short_cached" in sched_map and short_cached_admit_step is None:
                short_cached_admit_step = step_idx

            rids = list(sched_map)
            sampled = [[] if sched.requests[r].is_prefill_chunk else [13] for r in rids]
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=rids,
                    req_id_to_index={r: idx for idx, r in enumerate(rids)},
                    sampled_token_ids=sampled,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )

            if short_cold_done_step is None and len(short_cold.output_token_ids) >= 1:
                short_cold_done_step = step_idx
                short_cold_ttft_s = round(sim_clock_s - arrival_s, 3)
            if short_cached_done_step is None and len(short_cached.output_token_ids) >= 1:
                short_cached_done_step = step_idx
                short_cached_ttft_s = round(sim_clock_s - arrival_s, 3)
            if long1_done_step is None and len(long1.output_token_ids) >= 1:
                long1_done_step = step_idx
                long1_ttft_s = round(sim_clock_s, 3)

        cpu_us_per_step = (time.perf_counter() - t_cpu0) * 1e6 / max(step_idx, 1)
        aligned_ok = all(c == ALIGNED_CHUNK for c in long1_chunks[:-1])

        return {
            "arm": arm_name,
            "long_tokens": long_tokens,
            "short_cold_wait_steps": short_cold_admit_step,
            "short_cold_finish_steps": short_cold_done_step,
            "short_cold_ttft_s": short_cold_ttft_s,
            "short_cached_wait_steps": short_cached_admit_step,
            "short_cached_finish_steps": short_cached_done_step,
            "short_cached_ttft_s": short_cached_ttft_s,
            "long1_chunks_count": len(long1_chunks),
            "long1_boundaries_aligned": aligned_ok,
            "long1_done_step": long1_done_step,
            "long1_ttft_s": long1_ttft_s,
            "sched_cpu_us_per_step": round(cpu_us_per_step, 1),
        }

    results = []
    for long_len, label in [(65536, "64K"), (131072, "128K"), (262144, "256K")]:
        # 1. Baseline 1-lane FCFS (prefill_yield="off")
        results.append(
            run_scenario(
                f"1-lane FCFS ({label})",
                long_len,
                align_prefill_chunks=True,
                prefill_yield="off",
                max_parallel_prefills=1,
            )
        )
        # 2. Promoted 1107 1-lane Prefill Yield (prefill_yield="short")
        results.append(
            run_scenario(
                f"1-lane Prefill Yield ({label}, promoted 1107)",
                long_len,
                align_prefill_chunks=True,
                prefill_yield="short",
                max_parallel_prefills=1,
            )
        )
        # 3. 4-lane Aligned Whole-Chunk Rotate without Size Classes (patched aligned rotation)
        results.append(
            run_scenario(
                f"4-lane Aligned Size-Blind ({label}, patched rotation)",
                long_len,
                align_prefill_chunks=True,
                prefill_yield="off",
                max_parallel_prefills=4,
                prefill_long_threshold=0,
            )
        )
        # 4. 4-lane Aligned Whole-Chunk Rotate + Size-Classed QoS (prefill_yield="off")
        results.append(
            run_scenario(
                f"4-lane Aligned Size-Classed QoS, yield=off ({label})",
                long_len,
                align_prefill_chunks=True,
                prefill_yield="off",
                max_parallel_prefills=4,
                prefill_long_threshold=ALIGNED_CHUNK,
                max_long_prefills=1,
            )
        )
        # 5. 4-lane Unaligned Size-Blind
        results.append(
            run_scenario(
                f"4-lane Unaligned Size-Blind ({label})",
                long_len,
                align_prefill_chunks=False,
                prefill_yield="off",
                max_parallel_prefills=4,
                prefill_long_threshold=0,
            )
        )
        # 6. 4-lane Unaligned Size-Classed QoS (patched)
        results.append(
            run_scenario(
                f"4-lane Unaligned Size-Classed QoS ({label}, patched)",
                long_len,
                align_prefill_chunks=False,
                prefill_yield="off",
                max_parallel_prefills=4,
                prefill_long_threshold=ALIGNED_CHUNK,
                max_long_prefills=1,
            )
        )

    return {"scenarios": results}


def bench_engram_rows(tmp_dir: Path) -> dict:
    import numpy as np
    from vllm.models.deepseek_v4_1.engram_rows.rows import (
        EngramDiskRows,
        packed_table_bytes,
    )

    table_rows = 4096
    max_lookups = 24 * 64  # 64 tokens x 24 hash columns = 1536 lookups
    packed_file = tmp_dir / "engram_packed_bench.bin"
    nbytes = packed_table_bytes(table_rows)
    rng = np.random.default_rng(42)
    packed_file.write_bytes(rng.integers(1, 120, size=nbytes, dtype=np.uint8).tobytes())

    device = torch.device("cuda:0")
    stream = torch.cuda.current_stream(device)
    reader = EngramDiskRows(
        device=device,
        table_rows=table_rows,
        shard_start=0,
        shard_end=table_rows,
        max_lookups=max_lookups,
        cache_rows=1024,
    )
    try:
        reader.add_packed_source(str(packed_file), 0)
        hot_ids_np = np.arange(0, 24 * 16, dtype=np.int64) % 512
        reader.prefetch(hot_ids_np)
        time.sleep(0.02)

        hot_ids_gpu = torch.from_numpy(hot_ids_np).to(device=device)
        num_tokens_16 = torch.tensor([16], dtype=torch.int32, device=device)
        out_16 = torch.empty((16, 24 * 256), dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(10):
            reader.stage(stream, hot_ids_gpu, len(hot_ids_np), early=hot_ids_np)
            reader.gather(out_16, hot_ids_gpu.view(16, 24), num_tokens_16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(200):
            reader.stage(stream, hot_ids_gpu, len(hot_ids_np), early=hot_ids_np)
            reader.gather(out_16, hot_ids_gpu.view(16, 24), num_tokens_16)
        torch.cuda.synchronize()
        hot_us = (time.perf_counter() - t0) * 1e6 / 200.0

        # Decode-like path (early=None, GPU stream writes IDs + seq)
        t1 = time.perf_counter()
        for _ in range(200):
            reader.stage(stream, hot_ids_gpu, len(hot_ids_np), early=None)
            reader.gather(out_16, hot_ids_gpu.view(16, 24), num_tokens_16)
        torch.cuda.synchronize()
        decode_hot_us = (time.perf_counter() - t1) * 1e6 / 200.0

        stats = reader.stats()
    finally:
        reader.close()
        packed_file.unlink(missing_ok=True)

    return {
        "early_prefill_cached_stage_gather_us": round(hot_us, 2),
        "decode_stream_cached_stage_gather_us": round(decode_hot_us, 2),
        "reader_stats": stats,
    }


def main():
    boot_label = sys.argv[1] if len(sys.argv) > 1 else "boot1"
    tmp_dir = Path(f"/mnt/host-tmp/bench-{boot_label}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "boot": boot_label,
        "gpu": torch.cuda.get_device_name(0),
        "structured_output": bench_structured_output(tmp_dir),
        "scheduler": bench_scheduler(tmp_dir),
        "engram_rows": bench_engram_rows(tmp_dir),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
