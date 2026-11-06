# SPDX-License-Identifier: Apache-2.0
"""Experiment-only, eager target-route capture; disabled without a config file.

Snapshot the exact consumer tensors. Never pool invocations, repair IDs, infer
CED layout from row counts, or use CPU upper-bound query starts. This module is
mounted only in the separately identified capture arm; its timings are invalid
for performance comparison. Request content and token IDs are never saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def map_rows(
    ids,
    req_slots,
    positions,
    *,
    starts,
    full_positions,
    full_padding,
    request_names,
    prefill_lengths,
):
    """Map consumed rows to the GPU scheduler coordinates, including CED gathers."""
    nr = len(request_names)
    if len(starts) != nr + 1 or starts[0] != 0:
        raise ValueError("query-start shape/origin mismatch")
    if any(a > b for a, b in zip(starts, starts[1:])) or starts[-1] > len(
        full_positions
    ):
        raise ValueError("query starts outside allocated rows")
    if len(full_positions) != len(full_padding) or len(prefill_lengths) != nr:
        raise ValueError("full scheduler metadata shape mismatch")
    if not (len(ids) == len(req_slots) == len(positions)):
        raise ValueError("consumed rows and attention metadata differ")
    lookup = {}
    for request in range(nr):
        for row in range(starts[request], starts[request + 1]):
            key = (request, full_positions[row])
            if key in lookup:
                raise ValueError("duplicate scheduler request/position coordinate")
            lookup[key] = row
    result = []
    for selected, request, pos in zip(ids, req_slots, positions):
        live = [expert for expert in selected if expert >= 0]
        if len(live) != len(set(live)) or any(
            expert < -1 or expert >= 384 for expert in selected
        ):
            raise ValueError("invalid exact consumed expert IDs")
        if len(selected) != 6:
            raise ValueError("only native target top-6 capture is supported")
        original = lookup.get((request, pos))
        if original is None:
            if live or (request >= 0 and pos >= 0):
                raise ValueError("consumed row has no exact scheduler coordinate")
            result.append(
                {
                    "ids": selected,
                    "kind": "padding",
                    "request": None,
                    "position": pos,
                    "full_row": None,
                }
            )
            continue
        if live and full_padding[original]:
            raise ValueError("padding/dead scheduler row still routes live experts")
        if not live:
            kind = "dead"
        elif pos < prefill_lengths[request]:
            kind = "prefill"
        elif original == starts[request]:
            kind = "anchor"
        else:
            kind = "draft"
        result.append(
            {
                "ids": selected,
                "kind": kind,
                "request": request_names[request],
                "position": pos,
                "full_row": original,
            }
        )
    return result


class Capture:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text())
        for key in (
            "workload",
            "capture",
            "source_tree",
            "model_revision",
            "workload_sha256",
            "source_files_sha256",
            "output_dir",
            "layers",
        ):
            if not self.config.get(key):
                raise ValueError("capture config lacks " + key)
        for name, expected in self.config["source_files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("capture source mismatch: " + name)
        directory = Path(self.config["output_dir"])
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory / ("routes-pid-" + str(os.getpid()) + ".jsonl")
        self.handle = self.path.open("x")
        os.chmod(self.path, 0o600)
        self.step = 0
        self.count = 0
        self.bytes = 0
        self.phases = {"prefill": 0, "decode": 0, "mixed": 0}
        self.active = None
        self.names = {}
        self.scale_cache = {}
        self.pending = []
        self.limited = False
        self.summary()

    def summary(self):
        data = {
            "config": self.config,
            "records": self.count,
            "bytes": self.bytes,
            "steps_seen": self.step,
            "selected_steps_by_phase": self.phases,
            "limit_reached": self.limited,
            "timing_usable": False,
            "pid": os.getpid(),
        }
        self.path.with_suffix(".summary.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    def begin(self, batch, cg_mode, dummy, bonus_tokens, unsupported):
        if self.active is not None or self.pending:
            raise RuntimeError("previous capture step was not drained")
        if dummy:
            return
        self.step += 1
        if str(cg_mode).split(".")[-1] != "NONE":
            raise RuntimeError("route capture requires eager graph mode NONE")
        if unsupported or bonus_tokens != 1:
            raise RuntimeError(
                "capture does not support DBO/PCP/batch sharding or multi-anchor steps"
            )
        flags = [bool(v) for v in batch.is_prefilling_np[: batch.num_reqs]]
        phase = "prefill" if all(flags) else ("mixed" if any(flags) else "decode")
        limits = self.config.get(
            "steps_by_phase", {"prefill": 2, "mixed": 4, "decode": 16}
        )
        if self.limited or self.phases[phase] >= limits[phase]:
            return
        if batch.num_tokens_after_padding > self.config.get("max_rows", 16384):
            raise RuntimeError(
                "capture row budget exceeded; no invocation was truncated"
            )
        names = []
        for request in batch.req_ids:
            if request not in self.names:
                self.names[request] = "request-" + str(len(self.names))
            names.append(self.names[request])
        self.active = {
            "phase": phase,
            "request_names": names,
            "starts": batch.query_start_loc[: batch.num_reqs + 1].clone(),
            "full_positions": batch.positions.clone(),
            "full_padding": batch.is_padding.clone(),
            "prefill_lengths": [int(v) for v in batch.prefill_len_np[: batch.num_reqs]],
        }
        self.phases[phase] += 1

    def record(self, layer, attention_key, is_decoder, plan, ids, weights, packed):
        if self.active is None or plan.num_experts != 384:
            return
        match = re.fullmatch(r"(.+\.layers\.(\d+))\.ffn\.experts", layer)
        if match is None:
            raise RuntimeError("unrecognized target expert layer: " + layer)
        if int(match.group(2)) not in self.config["layers"]:
            return
        import torch

        from vllm.forward_context import get_forward_context

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("route capture cannot run during CUDA graph capture")
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            raise RuntimeError("microbatch metadata is unsupported")
        if attention_key not in metadata:
            raise RuntimeError("the layer exact attention metadata key is absent")
        original_metadata = metadata[attention_key]
        selected = original_metadata
        if is_decoder and selected.decoder is not None:
            selected = selected.decoder
        rows = ids.shape[0]
        compact = bool(is_decoder and selected is not original_metadata)
        expected_rows = (
            selected.num_actual_tokens
            if compact
            else self.active["full_positions"].numel()
        )
        if (
            rows != expected_rows
            or min(selected.positions.numel(), selected.req_id_per_token.numel()) < rows
        ):
            raise RuntimeError("consumer rows differ from selected attention metadata")
        if layer not in self.scale_cache:
            self.scale_cache[layer] = (packed[0][4].clone(), packed[1][4].clone())
        self.pending.append(
            {
                "layer": layer,
                "step": self.step,
                "intermediate": plan.intermediate,
                "decoder_layout": compact,
                "ids": ids.clone(),
                "weights": weights.clone(),
                "req_slots": selected.req_id_per_token[:rows].clone(),
                "positions": selected.positions[:rows].clone(),
            }
        )

    def end(self):
        if self.active is None:
            return
        active = self.active
        self.active = None
        # These copies synchronize after the target forward, once per selected
        # step. GPU snapshots preserve consumed values before reusable buffers
        # are overwritten. No route or scheduler tensor is ever mutated.
        starts = active["starts"].cpu().tolist()
        full_positions = active["full_positions"].cpu().tolist()
        full_padding = active["full_padding"].cpu().tolist()
        for item in self.pending:
            ids = item["ids"].cpu()
            weights = item["weights"].cpu()
            positions = item["positions"].cpu().tolist()
            req_slots = item["req_slots"].cpu().tolist()
            tp = 2304 // item["intermediate"]
            if tp not in (3, 4) or item["intermediate"] * tp != 2304:
                raise RuntimeError("unsupported target shard geometry")
            flags = self.scale_cache[item["layer"]]
            widths = {
                name: [2 if value & 1 else 4 for value in flag.cpu().tolist()]
                for name, flag in zip(("gate_up", "down"), flags)
            }
            record = {
                key: self.config[key]
                for key in (
                    "workload",
                    "capture",
                    "source_tree",
                    "model_revision",
                    "workload_sha256",
                )
            }
            record.update(
                evidence_kind="captured",
                step=item["step"],
                layer=item["layer"],
                phase=active["phase"],
                experts=384,
                topk=6,
                hidden=5120,
                tp=tp,
                decoder_layout=item["decoder_layout"],
                route_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                route_weights_sha256=hashlib.sha256(
                    weights.numpy().tobytes()
                ).hexdigest(),
                scale_bits={str(tp): widths},
                scheduler_query_starts=starts,
                rows=map_rows(
                    ids.tolist(),
                    req_slots,
                    positions,
                    starts=starts,
                    full_positions=full_positions,
                    full_padding=full_padding,
                    request_names=active["request_names"],
                    prefill_lengths=active["prefill_lengths"],
                ),
            )
            line = json.dumps(record, separators=(",", ":")) + "\n"
            size = len(line.encode())
            if self.count >= self.config.get(
                "max_records", 96
            ) or self.bytes + size > self.config.get("max_bytes", 67108864):
                self.limited = True
                break
            self.handle.write(line)
            self.handle.flush()
            self.count += 1
            self.bytes += size
        self.pending.clear()
        self.summary()


_CAPTURE = None
_CONFIG = os.environ.get("SPARK_EXPERT_ROUTE_CAPTURE_CONFIG")


def begin(batch, cg_mode, dummy, bonus_tokens, unsupported=False):
    global _CAPTURE
    if _CONFIG:
        if _CAPTURE is None:
            _CAPTURE = Capture(_CONFIG)
        _CAPTURE.begin(batch, cg_mode, dummy, bonus_tokens, unsupported)


def record(layer, attention_key, is_decoder, plan, ids, weights, packed):
    if _CAPTURE is not None:
        _CAPTURE.record(layer, attention_key, is_decoder, plan, ids, weights, packed)


def end():
    if _CAPTURE is not None:
        _CAPTURE.end()
