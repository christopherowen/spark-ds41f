"""Sonar, the benchmark client every measurement here runs through.

Sonar (pinned in config/sonar.json) is a separate repository: steady-state
load, the engine grid, model behaviour, prefill and stalls, suites and verdict tables, the
determinism gate, the quick probes and the measurement matrix. This deployment pins one
commit of it and keeps only what is its own: the suites sized to its shapes
(config/suites/), the A/A floors measured on its testbeds (scripts/sonar_table.py), the
chat template's reasoning switch (REASONING_OFF), and the conditions a bench runs under
(cooling, fan curve, memory and temperature watch, the deployment's identity).

The pinned commit is checked out once per host under .work/sonar/<revision>, from the pin's
URL, and verified there; nothing else is fetched. The pin is the checkout's own, not part of
an image or a release's upstreams lock: the benchmark client is the same for every arm a run
compares, whatever image or upstreams_config each arm has. SPARK_DS41F_SONAR names a local
Sonar checkout to use instead, for developing Sonar itself; a run records the revision it
used either way.

Imports stay inside functions: only commands that measure need Sonar, so cluster, doctor
and build commands never fetch it.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "sonar.json"
OVERRIDE_ENV = "SPARK_DS41F_SONAR"
CACHE = ROOT / ".work" / "sonar"
BOOT_DIR = ROOT / "scripts" / "_sonar_boot"
SONAR_EXT_ENV = "SPARK_DS41F_SONAR_EXT"
# Deployment suites (their digests are the recorded references' own); a name that is not
# one of them is a Sonar built-in suite or a JSON file.
SUITES = ROOT / "config" / "suites"
# The deployment's chat template turns reasoning off with this; Sonar sends nothing
# model-specific on its own.
REASONING_OFF = {"chat_template_kwargs": {"thinking": False}}
REASONING_OFF_JSON = json.dumps(REASONING_OFF)
# The bench client's stream collector, by the deployment's name for it (recorded in a run's
# identity as client_stream_policy) and Sonar's.
STREAM_POLICY_ENV = "SPARK_DS41F_BENCH_STREAM_POLICY"
STREAM_POLICIES = {"current": "current", "recorded-1030": "text-chunks"}
# The model barrier's variable before Sonar; Sonar's barrier reads SONAR_MODEL_CONTRACT with a
# contract of its own format. A run that still sets the old name is refused, not run unguarded.
LEGACY_CONTRACT_ENV = "SPARK_DS41F_SUITE_MODEL_CONTRACT"
# The schema names this repository's results were recorded under before Sonar (every run in
# manifests/suites/ and manifests/benchmarks/ through 1107): the same formats as Sonar's, so
# Sonar reads them as its own (sonar.add_schema_aliases; SONAR_SCHEMA_ALIASES for children).
RECORDED_SCHEMAS = {
    "servebench/1": "sonar-run/1",
    "servebench-engine/1": "sonar-engine/1",
    "servebench-model/1": "sonar-model/1",
    "servebench-prefill/1": "sonar-prefill/1",
    "servebench-stall/1": "sonar-stall/1",
    "servebench-suite/1": "sonar-suite/1",
    "servebench-model-evidence/1": "sonar-model-evidence/1",
}
RECORDED_SCHEMAS_JSON = json.dumps(RECORDED_SCHEMAS, sort_keys=True)


def legacy_contract_problem(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get(LEGACY_CONTRACT_ENV) or env.get(LEGACY_CONTRACT_ENV + "_SHA256"):
        return (f"{LEGACY_CONTRACT_ENV} is the model barrier from before Sonar, which Sonar does not read: write a "
                "sonar-model-evidence/1 contract and set SONAR_MODEL_CONTRACT and SONAR_MODEL_CONTRACT_SHA256 "
                "(Sonar's sonar/model_evidence.py), or run the experiment at its base commit")
    return None


class SonarUnavailable(RuntimeError):
    """The pinned Sonar commit could not be checked out."""


def pin(lock: Path = LOCK) -> dict:
    """The pin: upstream URL and full revision."""
    try:
        source = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SonarUnavailable(f"{lock}: {error}") from error
    if lock_problems(source):
        raise SonarUnavailable(f"{lock.relative_to(ROOT) if lock.is_relative_to(ROOT) else lock}: "
                               + "; ".join(lock_problems(source)))
    return source


def lock_problems(source: dict) -> list[str]:
    """What doctor reports about the pin (no network)."""
    problems = []
    if source.get("kind") != "git":
        problems.append("kind must be git")
    if not str(source.get("upstream", "")).startswith("https://"):
        problems.append("upstream must be an https URL")
    revision = str(source.get("revision", ""))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        problems.append("revision must be a full commit SHA")
    if not source.get("tracking_ref"):
        problems.append("tracking_ref is not recorded")
    return problems


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _verified(path: Path, revision: str) -> bool:
    try:
        return (_git("rev-parse", "HEAD", cwd=path) == revision
                and not _git("status", "--porcelain", "--untracked-files=no", cwd=path))
    except (OSError, subprocess.CalledProcessError):
        return False


# Checkouts this process has verified, by (cache, revision): verified once, not per call.
_VERIFIED: dict[tuple[str, str], Path] = {}


def checkout(lock: Path = LOCK) -> Path:
    """A verified checkout of the pinned commit (fetched on first use), or the override's."""
    override = os.environ.get(OVERRIDE_ENV)
    if override:
        path = Path(override).resolve()
        if not (path / "sonar" / "__main__.py").is_file():
            raise SonarUnavailable(f"{OVERRIDE_ENV}={override} is not a Sonar checkout")
        return path
    source = pin(lock)
    revision = source["revision"]
    path = CACHE / revision
    if _VERIFIED.get((str(CACHE), revision)) == path:
        return path
    if path.is_dir() and _verified(path, revision):
        _VERIFIED[(str(CACHE), revision)] = path
        return path
    if path.exists():
        raise SonarUnavailable(f"{path} is not a clean checkout of {revision}; remove it to fetch again")
    CACHE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{revision[:12]}-", dir=CACHE))
    try:
        _git("init", "-q", cwd=staging)
        _git("fetch", "-q", "--depth", "1", source["upstream"], revision, cwd=staging)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=staging)
        if not _verified(staging, revision):
            raise SonarUnavailable(f"{source['upstream']} did not give revision {revision}")
        try:
            staging.rename(path)
        except OSError:
            if not _verified(path, revision):  # another process put a different tree there
                raise
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise SonarUnavailable(f"cannot check out Sonar {revision[:12]} from {source['upstream']}: "
                               f"{str(detail).strip()}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    _VERIFIED[(str(CACHE), revision)] = path
    return path


def revision(path: Path | None = None) -> dict:
    """The Sonar a run used: its revision, whether it was the lock's, and a dirty flag."""
    path = path or checkout()
    head = _git("rev-parse", "HEAD", cwd=path) if (path / ".git").exists() else None
    pinned = pin()["revision"]
    return {"revision": head, "pinned": head == pinned,
            "dirty": bool(_git("status", "--porcelain", cwd=path)) if head else None}


DISTINCT_PROSE_PROMPTS: tuple[str, ...] = (
    "Explain, in precise technical prose, how speculative decoding increases language-model serving throughput while preserving the target output distribution. Cover acceptance rate, verification cost, batching, and latency.",
    "Explain, in precise technical prose, how continuous iteration-level batching interacts with paged KV-cache management in large transformer servers. Discuss memory fragmentation, preemption, and tail latency.",
    "Explain, in precise technical prose, how rotary positional embeddings scale to million-token contexts. Cover base frequency adjustment, YaRN interpolation, attention entropy, and extrapolation limits.",
    "Explain, in precise technical prose, how Multi-Head Latent Attention reduces KV-cache footprint during long-context inference. Discuss low-rank key-value compression, matrix absorption, and arithmetic intensity.",
    "Explain, in precise technical prose, how tensor parallelism partitions linear layers and self-attention across GPUs. Discuss Megatron column-row sharding, all-reduce synchronization, and communication hiding.",
    "Explain, in precise technical prose, why unified-memory GPU architectures behave differently from discrete-VRAM servers during small-batch decode. Cover L2 cache pollution, DMA prefetching, and memory bandwidth saturation.",
    "Explain, in precise technical prose, how Sparse Mixture-of-Experts routing affects decoding efficiency across nodes. Discuss top-k gating, expert load balance, irregular GEMM shapes, and all-to-all traffic.",
    "Explain, in precise technical prose, how block-wise FP8 weight and activation quantization preserves accuracy while accelerating GEMMs. Cover per-block scaling exponents, accumulator precision, and outlier handling.",
    "Explain, in precise technical prose, how CUDA Graphs reduce host launch overhead for latency-sensitive decode loops. Cover graph capture invariants, static buffer aliasing, shape bucketing, and replay dispatch.",
    "Explain, in precise technical prose, how sliding-window attention and recurrent state layers bound per-token state growth. Discuss fixed-size ring buffers, eviction semantics, and hybrid full-attention interleaved layers.",
    "Explain, in precise technical prose, how prefix caching reuses computed key-value blocks across multi-turn prompts. Cover block content hashing, reference counting, copy-on-write, and LRU eviction.",
    "Explain, in precise technical prose, how disaggregated prefill and decode workers avoid compute-versus-memory interference. Discuss KV-cache transfer over RDMA, time-to-first-token isolation, and queue scheduling.",
    "Explain, in precise technical prose, why vocabulary projection dominates single-stream speculative verification cost on wide vocabularies. Cover logit all-gather volume, top-k screening, and BF16 versus INT8 fidelity.",
    "Explain, in precise technical prose, how adaptive speculative verification budgets draft tokens dynamically under concurrency. Cover marginal step cost, token survival probability, and Tensor Core tile knees.",
    "Explain, in precise technical prose, how RDMA GPUDirect over convergent RoCE fabrics cuts collective latency on multi-node clusters. Cover queue-pair posting, inline headers, zero-copy registration, and ring versus direct topologies.",
    "Explain, in precise technical prose, how thermal throttling silently distorts long-context prefill benchmarks. Cover clock-frequency droop, inter-run cooling floors, sample variance, and sustained power limits.",
)
DISTINCT_CODE_PROMPTS: tuple[str, ...] = (
    "Write a complete Python implementation of an LRU cache using only the standard library. Include type hints, O(1) get and put methods, capacity validation, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a fixed-capacity ring buffer queue using only the standard library. Include type hints, push, pop, peek, overwrite-oldest handling, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a prefix trie supporting insert, search, starts_with, and delete using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of an LFU cache using only the standard library. Include type hints, O(1) get and put methods, capacity validation, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a disjoint-set (Union-Find) structure with path compression and union by rank using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a token-bucket rate limiter using only the standard library. Include type hints, monotonic clock refills, thread safety, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a segment tree supporting range sum queries and point updates using only the standard library. Include type hints, bounds checking, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a consistent hashing ring with virtual nodes using only the standard library. Include type hints, node add/remove, key lookup, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of an indexed binary min-heap supporting decrease_key and remove_key in O(log n) using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a directed graph topological sort with cycle detection (Kahn's algorithm) using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of an interval tree or sorted disjoint-interval merger supporting insert and overlap queries using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a Fenwick (Binary Indexed) tree supporting prefix sums and point updates using only the standard library. Include type hints, validation, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a thread-safe bounded blocking queue using only the standard library. Include type hints, optional timeouts on put and get, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of an adaptive Huffman or streaming run-length codec using only the standard library. Include type hints, round-trip checks, and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a skip-list ordered map supporting get, put, and delete using only the standard library. Include type hints and a short executable example. Explain the complexity after the code.",
    "Write a complete Python implementation of a deterministic Bloom filter with configurable bit size and hash count using only the standard library. Include type hints, validation, and a short executable example. Explain the complexity after the code.",
)


def _patch_matrix(matrix) -> None:
    if getattr(matrix, "_spark_ds41f_ext_applied", False):
        return
    matrix.DISTINCT_PROSE_PROMPTS = DISTINCT_PROSE_PROMPTS
    matrix.DISTINCT_CODE_PROMPTS = DISTINCT_CODE_PROMPTS
    matrix.DECODE_CASES["prose-distinct"] = matrix.DecodeCase(DISTINCT_PROSE_PROMPTS, per_stream=True)
    matrix.DECODE_CASES["code-distinct"] = matrix.DecodeCase(DISTINCT_CODE_PROMPTS, per_stream=True)
    matrix.DECODE_CASE_GROUPS["distinct"] = ["prose-distinct", "code-distinct"]
    matrix.Sampler.cool = None

    def cool_phase(self, phase: str) -> list[dict]:
        """Cool cluster nodes before a benchmark phase when a hook is configured."""
        if getattr(self, "cool", None) is None:
            return []
        return self.cool(phase)

    matrix.Sampler.cool_phase = cool_phase

    def _patched_suite_prefill(bench, options: argparse.Namespace, rng: random.Random) -> dict:
        """Cold prefill: unique uncached prompts, one output token, cooled per sample when enabled."""
        prefill_text = {"filler": matrix.filler, "novel": matrix.novel_text, "source": matrix.source_text}[options.prefill_text]
        sizes = [size for size in options.prefill_sizes if int(size * 0.97) < options.max_model_len - 64]
        samples: dict[str, list[dict]] = {str(size): [] for size in sizes}
        cool_fn = getattr(bench, "cool_phase", None) if getattr(bench, "cool", None) is not None else None
        for repeat in range(options.prefill_repeats):
            for size in rng.sample(sizes, len(sizes)):
                if cool_fn is not None:
                    if hasattr(bench, "wait_idle"):
                        bench.wait_idle()
                    cool_fn(f"prefill:{size}:r{repeat + 1}")
                payload = {
                    "model": options.model,
                    "prompt": prefill_text(int(size * 0.97), rng.randrange(1 << 30)) + "\n\nReply with the word ok.",
                    "max_tokens": 1,
                    "temperature": 0,
                    "cache_salt": uuid.uuid4().hex,
                }
                request = bench.run(f"prefill {size}", [("/v1/completions", payload)])["requests"][0]
                samples[str(size)].append(matrix.compact([request])[0])
                if request["ok"]:
                    print(f"  prefill {size}: {request['prompt_tokens'] / request['ttft_s']:.0f} tok/s", flush=True)
        summary = {}
        for size, requests in samples.items():
            ok = [request for request in requests if request["ok"]]
            summary[size] = {
                "prefill_tps": matrix.describe([request["prompt_tokens"] / request["ttft_s"] for request in ok]),
                "ttft_s": matrix.describe([request["ttft_s"] for request in ok]),
                "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
                "failed_requests": len(requests) - len(ok),
            }
        return {"points": summary, "samples": samples}

    def _patched_run_suites(sampler, options: argparse.Namespace, suites: list[str], report: dict, save) -> None:
        rng = random.Random(options.seed)
        cool_enabled = getattr(sampler, "cool", None) is not None
        ran_suite = False
        for suite_item in matrix.SUITES:
            if suite_item not in suites:
                continue
            if ran_suite and suite_item != "prefill" and cool_enabled:
                sampler.wait_idle()
                sampler.cool_phase(f"suite:{suite_item}")
            print(f"{suite_item}:", flush=True)
            report["suites"][suite_item] = matrix.SUITE_FUNCTIONS[suite_item](sampler, options, rng)
            save()
            if suite_item == "quality" and not report["suites"][suite_item]["ok"]:
                raise matrix.Abort("the quality gate failed; performance numbers would be meaningless")
            ran_suite = True

    matrix.suite_prefill = _patched_suite_prefill
    matrix.SUITE_FUNCTIONS["prefill"] = _patched_suite_prefill
    matrix.run_suites = _patched_run_suites
    matrix._spark_ds41f_ext_applied = True


def _patch_suite_and_prefill(suite_mod, prefill_mod) -> None:
    if not getattr(suite_mod, "_spark_ds41f_ext_applied", False):
        def cool_if_configured(phase: str, log=print) -> None:
            """Cool cluster nodes when ``SPARK_BENCH_COOL`` was exported by ``bin/spark bench``."""
            raw = os.environ.get("SPARK_BENCH_COOL")
            if not raw:
                return
            spec = json.loads(raw)
            below_c = float(spec["cool_below"])
            if below_c <= 0:
                return
            import spark_ds41f.bench as sb
            import spark_ds41f.config as sc
            cluster, nodes, _ = sc.configuration(
                argparse.Namespace(cluster_config=spec["cluster_config"], shape=None, nodes_config=None)
            )
            log(f"sonar: cooling before {phase} (< {below_c:g} C)")
            sb.cool_nodes(
                nodes,
                nodes["nodes"],
                below_c,
                float(spec["cool_timeout"]),
                fan_control=sb.fan_control_enabled(cluster),
            )

        def _patched_suite_run(main, url: str, suite: dict, out_dir: str, *, label: str | None = None,
                               corpus: str | None = None, metadata: dict | None = None, log=print,
                               model_evidence: dict | None = None) -> dict:
            """Run every step through sonar's ``main`` with optional inter-step cooling."""
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            me_api = suite_mod.model_evidence_api
            prepared_model = None
            if model_evidence and (suite["name"] == model_evidence["contract"]["suite"]
                                   or any(step["command"] == "model" for step in suite["steps"])):
                prior = [out / "suite.json", out / "model.json.gz", out / "model.json.gz.evidence.json",
                         out / "model.preflight.evidence.json"]
                if any(path.exists() for path in prior):
                    raise FileExistsError("prior opted-in model evidence exists; preserve it and use a distinct run")
                try:
                    prepared_model = me_api.prepare(model_evidence, suite, suite_mod.digest(suite), label, metadata)
                except Exception as error:
                    with (out / "model.preflight.evidence.json").open("x") as stream:
                        json.dump({"schema": me_api.SCHEMA, "status": "preflight-failed",
                            "contract": {"path": model_evidence["path"], "sha256": model_evidence["sha256"]},
                            "model_requests_started": False, "parser_worker": getattr(error, "receipt", None),
                            "errors": [f"{type(error).__name__}: {error}"]}, stream, indent=2)
                        stream.write("\n")
                    raise
            index = {"schema": suite_mod.SCHEMA, "suite": suite, "suite_sha256": suite_mod.digest(suite),
                     "label": label, "corpus": corpus,
                     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     "steps": []}
            if metadata is not None:
                index["metadata"] = metadata
            for step_idx, step in enumerate(suite["steps"]):
                if step_idx > 0 and step["command"] != "prefill":
                    suite_mod.cool_if_configured(f"step:{step['id']}", log=log)
                output = str(out / suite_mod.output_name(step))
                began = os.times().elapsed
                log(f"sonar suite {suite['name']}: step {step['id']} ({step['command']})")
                try:
                    code = main(suite_mod.arguments(step, url, output, label, corpus))
                except Exception as error:  # noqa: BLE001 - recorded; the remaining steps still run
                    code = f"{type(error).__name__}: {error}"[:300]
                index["steps"].append({"id": step["id"], "command": step["command"],
                                       "output": suite_mod.output_name(step),
                                       "exit": code, "seconds": round(os.times().elapsed - began, 1)})
                (out / "suite.json").write_text(json.dumps(index, indent=1) + "\n")
                if prepared_model is not None and step["command"] == "model":
                    try:
                        evidence = me_api.evaluate(output, code, prepared_model)
                    except Exception as error:
                        evidence = {"schema": me_api.SCHEMA, "status": "failed",
                            "contract": prepared_model["binding"], "process_exit": code,
                            "references": prepared_model.get("references", []),
                            "required_evidence_passed": False,
                            "errors": [f"{type(error).__name__}: {error}"]}
                    evidence_path = Path(output + ".evidence.json")
                    with evidence_path.open("x") as stream:
                        json.dump(evidence, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                    index["model_evidence"] = {"output": evidence_path.name,
                        "required_evidence_passed": evidence["required_evidence_passed"],
                        "contract_sha256": prepared_model["binding"]["sha256"]}
                    if not evidence["required_evidence_passed"]:
                        index["stopped_before_timing"] = True
                        log("required model evidence failed; remaining suite steps were not run")
                        break
            index["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            (out / "suite.json").write_text(json.dumps(index, indent=1) + "\n")
            return index

        suite_mod.cool_if_configured = cool_if_configured
        suite_mod.run = _patched_suite_run
        suite_mod._spark_ds41f_ext_applied = True

    if not getattr(prefill_mod, "_spark_ds41f_ext_applied", False):
        def _patched_prefill_run(base_url: str, model: str, sizes: list[int], *, repeats: int = 3, vllm: bool = True,
                                 extra_body: dict | None = None, log=lambda message: None) -> list[dict]:
            samples = []
            for repeat in range(repeats):
                for index, size in enumerate(sizes):
                    suite_mod.cool_if_configured(f"prefill:{size}:r{repeat + 1}", log=log)
                    text, counted = prefill_mod.document(base_url, model, size, seed=1000 * repeat + index, vllm=vllm)
                    body = {"model": model, "messages": [{"role": "user", "content": f"{text}\n\n{prefill_mod.QUESTION}"}],
                            "max_tokens": 1, "temperature": 0}
                    if vllm:
                        body["cache_salt"] = uuid.uuid4().hex
                    body.update(extra_body or {})
                    record = prefill_mod.client.stream_chat(base_url, body, continuous_usage=vllm,
                                                            timeout=max(900.0, size / 500))
                    ttft = record["first"] - record["sent"] if record["first"] is not None else None
                    tokens = record["prompt_tokens"] or counted
                    sample = {"size": size, "repeat": repeat, "prompt_tokens": tokens, "ttft_s": ttft,
                              "tps": tokens / ttft if tokens and ttft else None, "error": record["error"]}
                    samples.append(sample)
                    log(f"sonar: prefill {size} repeat {repeat + 1}/{repeats}: "
                        + (f"{sample['tps']:.0f} tok/s" if sample["tps"] else f"failed ({record['error']})"))
            return samples

        prefill_mod.run = _patched_prefill_run
        prefill_mod._spark_ds41f_ext_applied = True


def _patch_engine(engine_mod) -> None:
    if not getattr(engine_mod, "_spark_ds41f_ext_applied", False):
        orig_default_shapes = engine_mod.default_shapes
        orig_summarize = engine_mod.summarize
        orig_compare = engine_mod.compare
        orig_boot_ratios = engine_mod.boot_ratios
        orig_spread = engine_mod.spread

        def floor_samples(
            values: list[float],
            *,
            fraction: float = 0.25,
            min_count: int = 5,
        ) -> tuple[list[float], int]:
            """Fixed lower-order-statistic sample (fastest ``fraction`` of replays).

            Unlike ``median + 5 * MAD``, a fixed lower-quartile slice has no phase
            transition when asynchronous single-rank forward spikes affect 35-50% of
            replays in a 100-sample grid shape.
            """
            clean = sorted(v for v in values if v is not None)
            if not clean:
                return [], 0
            if len(clean) <= min_count:
                return clean, 0
            keep = min(len(clean), max(min_count, int(round(len(clean) * fraction))))
            return clean[:keep], len(clean) - keep

        def _patched_default_shapes(drafts: int = 5, max_context: int | None = None, prefill: bool = False,
                                    max_requests: int = 16, real_rows: bool = False) -> list[dict]:
            shapes = orig_default_shapes(drafts=drafts, max_context=max_context, prefill=prefill,
                                         max_requests=max_requests)
            if real_rows:
                shapes = [dict(shape, real_rows=True) for shape in shapes]
            return shapes

        def _patched_label(shape: dict) -> str:
            context = shape.get("context_len", 0)
            depth = f"{context // 1024}K" if context < 1 << 20 else f"{context / (1 << 20):.2g}M"
            real = " real" if shape.get("real_rows") else ""
            if shape.get("prefill"):
                return f"prefill {shape['num_tokens']}{real} @{depth}"
            return f"decode {shape.get('num_reqs', '?')}x{shape['num_tokens']}{real} rows @{depth}"

        def _patched_summarize(raw: dict) -> list[dict]:
            rows = orig_summarize(raw)
            for row, timed in zip(rows, raw["grid"]):
                kept_step, _ = floor_samples(timed["step_ms"])
                row["step_ms_floor"] = engine_mod.stats.describe(kept_step)
                if timed.get("forward_ms"):
                    kept_fwd, _ = floor_samples(timed["forward_ms"])
                    row["forward_ms_floor"] = engine_mod.stats.describe(kept_fwd)
            return rows

        def _patched_compare(base: dict, candidate: dict, *, tolerance_pct: float = 0.5,
                             confidence: float = 0.95, estimator: str = "5mad") -> dict:
            if estimator not in ("5mad", "floor25"):
                raise ValueError(f"unsupported engine grid estimator: {estimator}")
            import math
            by_label = {row["label"]: row for row in base["engine"]}
            if estimator == "5mad":
                out = orig_compare(base, candidate, tolerance_pct=tolerance_pct, confidence=confidence)
                floor_logs = []
                by_cand = {row["label"]: row for row in candidate["engine"]}
                for entry in out["rows"]:
                    metric = entry.get("metric")
                    other = by_label.get(metric)
                    row = by_cand.get(metric)
                    if other is None or row is None:
                        continue
                    fa, _ = floor_samples(other["samples"])
                    fb, _ = floor_samples(row["samples"])
                    if fa and fb:
                        ma, mb = sum(fa) / len(fa), sum(fb) / len(fb)
                        if ma:
                            chg = 100.0 * (mb / ma - 1.0)
                            entry["floor_base"] = ma
                            entry["floor_candidate"] = mb
                            entry["floor_change_pct"] = chg
                            if not row["shape"].get("prefill"):
                                floor_logs.append(math.log(mb / ma))
                if len(floor_logs) > 1:
                    gm_pct = 100.0 * (math.exp(sum(floor_logs) / len(floor_logs)) - 1.0)
                    out["floor_decode_gm_pct"] = gm_pct
                    for entry in out["rows"]:
                        if str(entry.get("metric", "")).startswith("decode geometric mean"):
                            entry["floor_change_pct"] = gm_pct
                return out

            pairs, warnings = [], []
            for row in candidate["engine"]:
                other = by_label.get(row["label"])
                if other is None:
                    warnings.append(f"{row['label']}: not in the base run")
                else:
                    pairs.append((other, row))
            family = 1 - (1 - confidence) / max(len(pairs), 1)
            rows, logs = [], []
            for other, row in pairs:
                a, dropped_a = floor_samples(other["samples"])
                b, dropped_b = floor_samples(row["samples"])
                test = engine_mod.stats.welch(a, b, confidence=family)
                if test is None:
                    continue
                ma, mb = sum(a) / len(a), sum(b) / len(b)
                rows.append({
                    "metric": row["label"], "base": ma, "candidate": mb,
                    "floor_base": ma, "floor_candidate": mb,
                    "change_pct": test["change_pct"], "floor_change_pct": test["change_pct"],
                    "ci95_pct": test["ci95_pct"], "trimmed": dropped_a + dropped_b,
                    "verdict": engine_mod.stats.verdict(test["change_pct"], test["ci95_pct"], False, tolerance_pct),
                })
                if not row["shape"].get("prefill"):
                    logs.append((math.log(1 + test["change_pct"] / 100), test["se_pct"] / 100))
            floor_gm = None
            if len(logs) > 1:
                mean = sum(l for l, _ in logs) / len(logs)
                se = math.sqrt(sum(s * s for _, s in logs)) / len(logs)
                change = 100 * (math.exp(mean) - 1)
                ci = 100 * engine_mod.stats.t_quantile(1 - (1 - confidence) / 2, 1e6) * se
                floor_gm = change
                rows.append({
                    "metric": f"decode geometric mean ({len(logs)})",
                    "base": None, "candidate": None,
                    "change_pct": change, "floor_change_pct": change, "ci95_pct": ci,
                    "verdict": engine_mod.stats.verdict(change, ci, False, tolerance_pct),
                })
            note = (f"intervals hold for all {len(pairs)} shapes at once ({100 * confidence:.0f}%, Bonferroni); "
                    f"fastest-25% order-statistic floor slice used (upper 75% dropped: "
                    f"{sum(r.get('trimmed', 0) for r in rows)})")
            out = {"warnings": warnings, "rows": rows, "note": note, "estimator": "floor25"}
            if floor_gm is not None:
                out["floor_decode_gm_pct"] = floor_gm
            return out

        def _patched_boot_ratios(results: list[dict], reference: dict, *, estimator: str = "5mad") -> list[float]:
            if estimator == "5mad":
                return orig_boot_ratios(results, reference)
            if estimator != "floor25":
                raise ValueError(f"unsupported engine grid estimator: {estimator}")
            import math
            def means(run: dict) -> dict:
                out = {}
                for row in run["engine"]:
                    if not row["shape"].get("prefill"):
                        kept, _ = floor_samples(row["samples"])
                        out[row["label"]] = sum(kept) / len(kept)
                return out
            base = means(reference)
            ratios = []
            for run in results:
                logs = [math.log(value / base[label]) for label, value in means(run).items() if label in base]
                ratios.append(math.exp(sum(logs) / len(logs)) if logs else float("nan"))
            return ratios

        def _patched_spread(runs: list[dict], against: list[dict] | None = None,
                            *, estimator: str = "5mad") -> dict:
            if estimator == "5mad":
                return orig_spread(runs, against)
            if estimator != "floor25":
                raise ValueError(f"unsupported engine grid estimator: {estimator}")
            import math
            ratios = _patched_boot_ratios(runs, runs[0], estimator="floor25")
            middle = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            pct = [100 * (r / middle - 1) for r in ratios]
            out = {"boots": [{"label": run.get("label"), "vs_mean_pct": round(v, 3)} for run, v in zip(runs, pct)],
                   "sd_pct": engine_mod.stats.describe(pct).get("sd"), "range_pct": max(pct) - min(pct),
                   "estimator": "floor25"}
            if against:
                others = [100 * (r / middle - 1) for r in _patched_boot_ratios(against, runs[0], estimator="floor25")]
                out["against"] = {"boots": [{"label": run.get("label"), "vs_mean_pct": round(v, 3)}
                                            for run, v in zip(against, others)],
                                  "sd_pct": engine_mod.stats.describe(others).get("sd")}
                mean_a, mean_b = sum(pct) / len(pct), sum(others) / len(others)
                test = engine_mod.stats.welch([100 + v for v in pct], [100 + v for v in others])
                out["difference"] = {"change_pct": mean_b - mean_a, "ci95_pct": test["ci95_pct"] if test else None,
                                     "verdict": engine_mod.stats.verdict(
                                         mean_b - mean_a, test["ci95_pct"] if test else None, False, 0.0)}
            return out

        def _patched_engine_run(base_url: str, shapes: list[dict], *, replays: int = 100, warmup: int = 2,
                                settle: int = 1, timeout: float = 3600.0, require_idle: bool = True,
                                fill_real_context: bool = True) -> dict:
            base = base_url.rstrip("/")
            if require_idle:
                engine_mod.prom.require_idle(base, "the grid")
            spec = {"shapes": [{k: v for k, v in s.items() if k != "kind"} for s in shapes],
                    "replays": replays, "warmup": warmup, "settle": settle}
            if not fill_real_context:
                spec["fill_real_context"] = False
            orig_error: BaseException | None = None
            reset_error: Exception | None = None
            try:
                answer = engine_mod._post(base + "/collective_rpc",
                                          {"method": "engine_grid", "args": [json.dumps(spec)], "timeout": timeout},
                                          timeout + 60)
                workers = (answer or {}).get("results") or []
                grids = [w for w in workers if isinstance(w, dict)]
                if not grids:
                    raise RuntimeError(f"no engine grid in the reply: {str(answer)[:300]}")
                rank0 = min(grids, key=lambda g: g.get("rank", 0))
                ignored = [engine_mod.label(shape) for shape, timed in zip(shapes, rank0["shapes"])
                           if shape.get("real_rows") and not timed.get("real_rows")]
                if ignored:
                    raise RuntimeError(f"the server ran padding rows for {', '.join(ignored)}: "
                                       "its engine grid has no real_rows")
            except BaseException as error:
                orig_error = error
                raise
            finally:
                try:
                    engine_mod._post(base + "/reset_prefix_cache", None, 120)
                except Exception as error:  # noqa: BLE001 - fail or preserve original error
                    reset_error = error
                    if orig_error is not None and hasattr(orig_error, "add_note"):
                        orig_error.add_note(
                            f"prefix cache reset failed after engine grid error: {type(error).__name__}: {error}"
                        )
            if reset_error is not None:
                raise RuntimeError(
                    f"prefix cache reset failed after engine grid: {type(reset_error).__name__}: {reset_error}"
                ) from reset_error
            return {"workers": len(workers), "rank": rank0.get("rank"), "prefix_cache_reset": True,
                    "replays": replays, "warmup": warmup, "settle": rank0.get("settle"), "shapes": shapes,
                    "grid": rank0["shapes"]}

        engine_mod.floor_samples = floor_samples
        engine_mod.default_shapes = _patched_default_shapes
        engine_mod.label = _patched_label
        engine_mod.summarize = _patched_summarize
        engine_mod.compare = _patched_compare
        engine_mod.boot_ratios = _patched_boot_ratios
        engine_mod.spread = _patched_spread
        engine_mod.run = _patched_engine_run
        engine_mod._spark_ds41f_ext_applied = True


def patch_main_namespace(ns: dict) -> None:
    """Wrap ``build_parser`` and ``command_engine`` inside ``sonar.__main__`` / ``__main__``."""
    if ns.get("_spark_ds41f_ext_applied"):
        return
    orig_build_parser = ns["build_parser"]
    engine_mod = ns["engine"]
    version = ns["__version__"]
    write_json = ns["write_json"]

    def _patched_command_engine(args: argparse.Namespace) -> int:
        real_rows = getattr(args, "real_rows", False)
        fill_real_context = getattr(args, "fill_real_context", True)
        if args.grid == "default":
            longest = args.max_context or engine_mod.max_model_len(args.url)
            shapes = engine_mod.default_shapes(drafts=args.drafts, max_context=longest,
                                               prefill=args.prefill, max_requests=args.max_requests,
                                               real_rows=real_rows)
        else:
            with open(args.grid, encoding="utf-8") as handle:
                shapes = json.load(handle)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        raw = engine_mod.run(args.url, shapes, replays=args.replays, warmup=args.warmup,
                             settle=args.settle, fill_real_context=fill_real_context)
        rows = engine_mod.summarize(raw)
        result = {"schema": engine_mod.SCHEMA,
                  "tool": {"name": "sonar", "version": version},
                  "started_utc": started, "label": args.label,
                  "server": {"url": args.url, "workers": raw["workers"], "rank": raw["rank"],
                             "prefix_cache_reset": raw["prefix_cache_reset"]},
                  "workload": {"grid": args.grid, "replays": args.replays, "warmup": args.warmup,
                               "settle": raw["settle"], "drafts": args.drafts, "prefill": args.prefill,
                               "max_requests": args.max_requests, "real_rows": real_rows},
                  "engine": rows}
        if args.output:
            write_json(args.output, result)
        for row in rows:
            d = row["step_ms"]
            print(f"{row['label']:32s} {d['mean']:9.3f} ms ±{d['ci95_pct'] or 0:.2f}%", file=sys.stderr)
        widest = max((row["step_ms"].get("ci95_pct") or 0) for row in rows)
        print(json.dumps({"summary": "sonar engine", "label": args.label, "shapes": len(rows),
                          "widest_ci95_pct": widest, "prefix_cache_reset": raw["prefix_cache_reset"],
                          "output": args.output}), flush=True)
        return 0 if raw["prefix_cache_reset"] else 1

    def _patched_build_parser() -> argparse.ArgumentParser:
        parser = orig_build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        grid = sub.choices["engine"]
        if "--real-rows" not in grid._option_string_actions:
            grid.add_argument(
                "--real-rows",
                action="store_true",
                help="run the default grid's rows as real ones: they route to experts and read "
                     "a context the server fills first (needs a server whose grid has real_rows)",
            )
            grid.add_argument(
                "--no-fill-real-context",
                dest="fill_real_context",
                action="store_false",
                help="skip refilling context before real_rows shapes (only immediately after a prior fill)",
            )
        grid.set_defaults(func=_patched_command_engine)
        return parser

    ns["command_engine"] = _patched_command_engine
    ns["build_parser"] = _patched_build_parser
    ns["_spark_ds41f_ext_applied"] = True


def install_main_hook() -> None:
    """Install a meta-path hook so ``python3 -m sonar`` patches ``__main__`` without pre-importing it."""
    if any(getattr(finder, "_spark_ds41f_main_hook", False) for finder in sys.meta_path):
        return

    class _SonarMainLoader:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def get_code(self, fullname: str):
            source = self._inner.get_source(fullname) if hasattr(self._inner, "get_source") else None
            if source is None:
                return self._inner.get_code(fullname)
            hook = "\nfrom spark_ds41f import sonar as _ds41f_sonar\n_ds41f_sonar.patch_main_namespace(globals())\n"
            marker = 'if __name__ == "__main__":'
            alt = "if __name__ == '__main__':"
            if marker in source:
                source = source.replace(marker, hook + "\n" + marker, 1)
            elif alt in source:
                source = source.replace(alt, hook + "\n" + alt, 1)
            else:
                source = source + "\n" + hook
            filename = self._inner.get_filename(fullname) if hasattr(self._inner, "get_filename") else fullname
            return compile(source, filename, "exec")

    class _SonarMainFinder:
        _spark_ds41f_main_hook = True

        def find_spec(self, fullname: str, path, target=None):
            if fullname != "sonar.__main__":
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _SonarMainLoader(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _SonarMainFinder())


def apply_deployment_extensions(*, include_main: bool = True) -> None:
    """Layer spark-ds41f's post-pin Sonar extensions (distinct decode cases, cooling hooks, real_rows)
    onto the checked-out Sonar modules in this process."""
    path = str(checkout())
    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.import_module("sonar").add_schema_aliases(RECORDED_SCHEMAS)
    _patch_matrix(importlib.import_module("sonar.matrix"))
    _patch_suite_and_prefill(importlib.import_module("sonar.suite"), importlib.import_module("sonar.prefill"))
    _patch_engine(importlib.import_module("sonar.engine"))
    if include_main:
        patch_main_namespace(importlib.import_module("sonar.__main__").__dict__)


def module(name: str = "sonar"):
    """Import ``sonar`` (or ``sonar.<name>``) from the pinned checkout."""
    path = str(checkout())
    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.import_module("sonar").add_schema_aliases(RECORDED_SCHEMAS)
    loaded = importlib.import_module(name if name == "sonar" or name.startswith("sonar.") else f"sonar.{name}")
    apply_deployment_extensions()
    return loaded


def environment(base: dict | None = None) -> dict:
    """An environment for ``python3 -m sonar``: the checkout on PYTHONPATH and the stream
    collector the deployment's policy names."""
    env = dict(os.environ if base is None else base)
    problem = legacy_contract_problem(env)
    if problem:
        raise SonarUnavailable(problem)
    env["PYTHONPATH"] = os.pathsep.join([str(checkout()), str(BOOT_DIR), str(ROOT), *filter(None, [env.get("PYTHONPATH")])])
    env[SONAR_EXT_ENV] = "1"
    policy = env.get(STREAM_POLICY_ENV, "current")
    if policy not in STREAM_POLICIES:
        raise ValueError(f"unsupported stream collector policy: {policy}")
    env["SONAR_STREAM_COLLECTOR"] = STREAM_POLICIES[policy]
    env["SONAR_SCHEMA_ALIASES"] = RECORDED_SCHEMAS_JSON
    return env


def stream_policy() -> str:
    """The deployment's name for the stream collector this run uses."""
    policy = os.environ.get(STREAM_POLICY_ENV, "current")
    if policy not in STREAM_POLICIES:
        raise ValueError(f"unsupported stream collector policy: {policy}")
    return policy


def argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "sonar", *args]


def run(*args: str, **kwargs) -> int:
    """Run a Sonar command in a child process; its exit code."""
    return subprocess.run(argv(*args), env=environment(), cwd=ROOT, **kwargs).returncode


def suite_name(name: str) -> str:
    """The suite argument Sonar gets: a deployment suite's file, else the name as given."""
    path = SUITES / f"{name}.json"
    return str(path) if path.is_file() else name


def load_suite(name: str) -> dict:
    return module("suite").load(suite_name(name))


def suite_arguments(args: list[str]) -> list[str]:
    """``--suite NAME`` (or ``--suite=NAME``) of config/suites/ becomes that file."""
    out = list(args)
    for index, value in enumerate(out):
        if value == "--suite" and index + 1 < len(out):
            out[index + 1] = suite_name(out[index + 1])
        elif value.startswith("--suite="):
            out[index] = "--suite=" + suite_name(value.split("=", 1)[1])
    return out


# The lean profile's mixed-traffic long prompts have always been this directory's sources.
MIXED_LONG_SOURCE = ROOT / "experiments" / "2026-09-29-determinism"


def mixed_arguments(args: list[str]) -> list[str]:
    if any(value == "--long-source" or value.startswith("--long-source=") for value in args):
        return list(args)
    return [*args, "--long-source", str(MIXED_LONG_SOURCE)]


def exec_command(command: str, args: list[str], *, reasoning_off: bool = False) -> int:
    """A lab shim: ``python3 scripts/sonar_<command>.py URL ARGS...`` runs the pinned Sonar's
    command, with the deployment's reasoning switch unless the arguments name their own."""
    if reasoning_off and "--extra-body" not in args:
        args = [*args, "--extra-body", REASONING_OFF_JSON]
    os.environ.update(environment())
    sys.path.insert(0, str(checkout()))
    apply_deployment_extensions()
    main = importlib.import_module("sonar.__main__").main
    return main([command, *args])
