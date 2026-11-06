"""CPU-only verification of VLLM_DS41_LSA default-off bit-identity, no downward quantization, and 32K-1M routing."""
import ast
import json
import os
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OVERLAY = EXP / "overlay"


def load_function(path: Path, func_name: str, scope: dict, class_name: str | None = None):
    tree = ast.parse(path.read_text())
    if class_name is not None:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    else:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    mod = ast.Module(body=[fn], type_ignores=[])
    local_scope = dict(scope)
    exec(compile(mod, str(path), "exec"), local_scope)
    return local_scope[func_name]


class DummyNativePlan:
    def __init__(self, width, page, *, candidates=False, candidate_blocks=False):
        self.kind = "native"
        self.width = width
        self.page = page
        self.candidates = candidates
        self.candidate_blocks = candidate_blocks

    def scratch_specs(self, rows):
        return (("native", rows, self.width),)

    def run(self, *args, **kwargs):
        return ("native", self.width, self.page, self.candidate_blocks)


class DummyShortlistBase:
    pass


class DummyQueryReusePlan(DummyShortlistBase):
    def __init__(self, width, page, *, candidate_blocks=False, budget=None, row_cap=256):
        self.kind = "block8_query_reuse"
        self.width = width
        self.page = page
        self.candidate_blocks = candidate_blocks
        self.budget = budget or (512 if candidate_blocks else 64)
        self.block_budget = self.budget * 8
        self.row_cap = row_cap
        self.coarse_kind = (
            "RawWideCoarseTopK"
            if self.block_budget == 8192 and width // 8 > 8192
            else "DirectCoarseTopK"
            if self.block_budget == 4096 and width // 8 > 4096
            else "IndexerTopK"
        )

    def scratch_specs(self, rows):
        return (("shortlist", min(rows, self.row_cap), self.width, self.budget),)

    def run(self, *args, **kwargs):
        return ("shortlist", self.width, self.page, self.budget, self.coarse_kind)


def check_ast_invariants():
    attn_text = (OVERLAY / "attention.py").read_text()
    model_attn_text = (OVERLAY / "model_attention.py").read_text()
    tl_attn_tree = ast.parse(attn_text)
    imported_modules = [
        n.module for n in ast.walk(tl_attn_tree) if isinstance(n, ast.ImportFrom) and n.module
    ]
    assert "compact_shortlist" not in imported_modules
    assert "direct_shortlist" not in imported_modules
    for legacy in (
        "VLLM_DS41_PAGE_SHORTLIST",
        "VLLM_DS41_PAGE_MODE",
        "VLLM_DS41_PAGE_NATIVE_DECODE",
        "VLLM_DS41_PAGE_NATIVE_SOURCE",
        "VLLM_DS41_PAGE_SOURCE_BUDGET",
        "VLLM_DS41_PAGE_DENSE_BUDGET",
        "VLLM_DS41_PAGE_ROWS",
        "VLLM_DS41_PAGE_DECODE_MIN_ROWS",
        "compact8",
        "page64",
    ):
        assert legacy not in attn_text, f"{legacy} still present in attention.py"
        assert legacy not in model_attn_text, f"{legacy} still present in model_attention.py"

    bs_text = (OVERLAY / "block_shortlist.py").read_text()
    qr_text = (OVERLAY / "query_reuse.py").read_text()
    si_text = (OVERLAY / "shortlist_indexer.py").read_text()
    for forbidden in ("_e2m1_code", "_mxfp4_byte", "quantize_mxfp4", "CompactShortlistPlan", "summarize_compact"):
        assert forbidden not in bs_text, f"{forbidden} found in block_shortlist.py"
        assert forbidden not in qr_text, f"{forbidden} found in query_reuse.py"

    bs_tree = ast.parse(bs_text)
    qr_tree = ast.parse(qr_text)
    si_tree = ast.parse(si_text)

    def has_pdl_chain(tree, fn_name):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "pdl"
                and call.func.attr == "chain"
            ):
                return True
        return False

    assert has_pdl_chain(bs_tree, "summarize_blocks")
    assert has_pdl_chain(bs_tree, "block_scores")
    assert has_pdl_chain(bs_tree, "block_candidates")
    assert has_pdl_chain(qr_tree, "prepared_scores")
    assert has_pdl_chain(qr_tree, "prepare_queries")
    assert has_pdl_chain(si_tree, "remap_blocks")

    sb_fn = next(n for n in bs_tree.body if isinstance(n, ast.FunctionDef) and n.name == "summarize_blocks")
    threads_vals = [
        kw.value.value
        for call in ast.walk(sb_fn)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg == "threads" and isinstance(kw.value, ast.Constant)
    ]
    assert threads_vals == [512], f"Expected threads=512 in summarize_blocks, got {threads_vals}"
    return {
        "single_env_knob": "VLLM_DS41_LSA",
        "no_legacy_page_env_vars": True,
        "no_compact_imports_in_attention": True,
        "no_downward_quantization_in_block8": True,
        "summarize_blocks_threads": 512,
        "all_pipeline_kernels_pdl_chained": True,
    }


def build_dispatch_harness():
    attn_path = OVERLAY / "attention.py"
    model_attn_path = OVERLAY / "model_attention.py"

    scope = {
        "os": os,
        "functools": __import__("functools"),
        "SimpleNamespace": types.SimpleNamespace,
        "STATE_BYTES": 100,
        "summarize_blocks": "summarize_blocks",
        "BlockMeanShortlistPlan": DummyQueryReusePlan,
        "PageShortlistPlan": DummyShortlistBase,
        "IndexerV41Plan": DummyNativePlan,
        "current_workspace_manager": lambda: types.SimpleNamespace(
            get_simultaneous=lambda *specs: list(specs)
        ),
        "retain_cuda_graph_capture_resource": lambda *args: None,
    }
    for fn_name in (
        "_lsa_enabled",
        "_shortlist_options",
        "_source_short_options",
        "_index_plan",
        "_source_short_plan",
        "_index_chunk",
    ):
        cls_name = "DeepseekV41TileLangAttention" if fn_name == "_index_chunk" else None
        scope[fn_name] = load_function(attn_path, fn_name, scope, class_name=cls_name)

    model_scope = {
        "os": os,
        "re": __import__("re"),
        "torch": types.SimpleNamespace(uint8="uint8"),
        "ced_decoder_start": lambda hf: None,
        "SlidingWindowMLASpec": lambda **kw: kw,
        "MLAAttentionSpec": lambda **kw: kw,
    }
    model_scope["get_kv_cache_spec"] = load_function(
        model_attn_path, "get_kv_cache_spec", model_scope, class_name="_Cache"
    )
    return scope, model_scope


def test_default_off(scope, model_scope):
    for env_val in (None, "0"):
        os.environ.pop("VLLM_DS41_LSA", None)
        if env_val is not None:
            os.environ["VLLM_DS41_LSA"] = env_val
        scope["_index_plan"].cache_clear()
        scope["_source_short_plan"].cache_clear()

        dummy_cache = types.SimpleNamespace(block_size=128, kind="index", ratio=2, window=0, draft=False, prefix="layers.2")
        spec = model_scope["get_kv_cache_spec"](dummy_cache, None)
        assert spec["state_content_bytes"] == 68, f"Expected 68 bytes when off, got {spec['state_content_bytes']}"
        assert not scope["_lsa_enabled"]()
        for args in [(16384, 128, False, False), (524288, 128, False, False), (1048576, 256, False, True)]:
            plan = scope["_index_plan"](*args)
            assert isinstance(plan, DummyNativePlan)
            assert scope["_source_short_plan"](*args) is None
    return {"default_off_state_bytes": 68, "default_off_all_native_plans": True}


def test_opt_in_routing(scope, model_scope):
    os.environ["VLLM_DS41_LSA"] = "1"
    scope["_index_plan"].cache_clear()
    scope["_source_short_plan"].cache_clear()

    dummy_cache = types.SimpleNamespace(block_size=128, kind="index", ratio=2, window=0, draft=False, prefix="layers.2")
    spec = model_scope["get_kv_cache_spec"](dummy_cache, None)
    assert spec["state_content_bytes"] == 100

    dense_short = scope["_index_plan"](16384, 128, False, False)
    dense_full = scope["_index_plan"](524288, 128, False, False)
    assert isinstance(dense_short, DummyQueryReusePlan) and dense_short.budget == 64 and dense_short.coarse_kind == "IndexerTopK"
    assert isinstance(dense_full, DummyQueryReusePlan) and dense_full.budget == 64 and dense_full.coarse_kind == "IndexerTopK"

    source_full = scope["_index_plan"](1048576, 256, False, True)
    source_short = scope["_source_short_plan"](1048576, 256, False, True)
    assert isinstance(source_full, DummyQueryReusePlan) and source_full.budget == 1024 and source_full.coarse_kind == "RawWideCoarseTopK"
    assert isinstance(source_short, DummyQueryReusePlan) and source_short.budget == 512 and source_short.coarse_kind == "DirectCoarseTopK"

    def make_recording_plan(plan, label):
        p = types.SimpleNamespace(
            label=label,
            kind=plan.kind,
            width=plan.width,
            page=plan.page,
            budget=getattr(plan, "budget", None),
            coarse_kind=getattr(plan, "coarse_kind", None),
            scratch_specs=plan.scratch_specs,
        )
        return p

    def wrap_shortlist(plan, label):
        class Wrapped(DummyShortlistBase):
            def __init__(self):
                self.label = label
                self.kind = plan.kind
                self.width = plan.width
                self.page = plan.page
                self.budget = plan.budget
                self.coarse_kind = plan.coarse_kind
                self.scratch_specs = plan.scratch_specs

        return Wrapped()

    dense_full = wrap_shortlist(dense_full, "shortlist_full_64")
    dense_short = wrap_shortlist(dense_short, "shortlist_short_64")
    source_full = wrap_shortlist(source_full, "shortlist_source_1024")
    source_short = wrap_shortlist(source_short, "source_short_512")

    def run_dispatch(layer_plans, mode, rows, score_width):
        called = {}
        for d in (layer_plans.index, layer_plans.native_index):
            for k, v in list(d.items()):
                if not hasattr(v, "label"):
                    d[k] = make_recording_plan(v, f"native_{k}")
                v_obj = d[k]
                v_obj.run = lambda *a, _v=v_obj, **kw: called.update(
                    label=_v.label, width=_v.width, budget=_v.budget, coarse=_v.coarse_kind
                )
        if layer_plans.source_short is not None:
            ss = layer_plans.source_short
            ss.run = lambda *a, _v=ss, **kw: called.update(
                label=_v.label, width=_v.width, budget=_v.budget, coarse=_v.coarse_kind
            )
        dummy_self = types.SimpleNamespace(
            _tilelang=lambda: layer_plans,
            indexer=types.SimpleNamespace(k_cache=types.SimpleNamespace(kv_cache="kv")),
            _active="active",
        )
        scope["_index_chunk"](
            dummy_self,
            mode,
            rows,
            output_indices="out",
            q_mxfp4="q",
            q_scales="s",
            query_weights="w",
            page_table="pt",
            cache_lengths="cl",
            score_width=score_width,
        )
        return called

    dense_plans = types.SimpleNamespace(
        index={"decode": dense_full, "prefill": dense_full, "prefill_short": dense_short},
        native_index={
            "decode": DummyNativePlan(524288, 128),
            "prefill": DummyNativePlan(524288, 128),
            "prefill_short": DummyNativePlan(16384, 128),
        },
        source_short=None,
    )
    source_plans = types.SimpleNamespace(
        index={"decode": source_full, "prefill": source_full},
        native_index={
            "decode": DummyNativePlan(1048576, 256, candidate_blocks=True),
            "prefill": DummyNativePlan(1048576, 256, candidate_blocks=True),
        },
        source_short=source_short,
    )

    prefill_routes = {}
    decode_r8_routes = {}
    for tokens in (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576):
        dense_sw = tokens // 2
        dense_mode = "prefill_short" if dense_sw <= 16384 else "prefill"
        d_res = run_dispatch(dense_plans, dense_mode, 64, dense_sw)
        s_res = run_dispatch(source_plans, "prefill", 64, tokens)
        prefill_routes[str(tokens)] = {"dense_layers_2_8_14": d_res, "source_layer_20": s_res}

        dense_dec_mode = "prefill_short" if dense_sw <= 16384 else "decode"
        d_dec_res = run_dispatch(dense_plans, dense_dec_mode, 8, dense_sw)
        s_dec_res = run_dispatch(source_plans, "decode", 8, tokens)
        decode_r8_routes[str(tokens)] = {"dense_layers_2_8_14": d_dec_res, "source_layer_20": s_dec_res}

    assert prefill_routes["4096"]["dense_layers_2_8_14"]["label"] == "native_prefill_short"
    assert prefill_routes["8192"]["dense_layers_2_8_14"]["label"] == "native_prefill_short"
    assert prefill_routes["16384"]["dense_layers_2_8_14"]["label"] == "shortlist_short_64"
    assert prefill_routes["32768"]["dense_layers_2_8_14"]["label"] == "shortlist_short_64"
    assert prefill_routes["32768"]["source_layer_20"]["label"] == "native_prefill"
    assert decode_r8_routes["32768"]["dense_layers_2_8_14"]["label"] == "shortlist_short_64"
    assert decode_r8_routes["32768"]["source_layer_20"]["label"] == "native_decode"

    assert prefill_routes["65536"]["dense_layers_2_8_14"]["label"] == "shortlist_full_64"
    assert prefill_routes["65536"]["source_layer_20"]["label"] == "source_short_512"
    assert decode_r8_routes["65536"]["dense_layers_2_8_14"]["label"] == "shortlist_full_64"
    assert decode_r8_routes["65536"]["source_layer_20"]["label"] == "source_short_512"

    for tok in ("131072", "262144", "524288", "1048576"):
        assert prefill_routes[tok]["dense_layers_2_8_14"]["label"] == "shortlist_full_64"
        assert prefill_routes[tok]["source_layer_20"]["label"] == "shortlist_source_1024"
        assert decode_r8_routes[tok]["dense_layers_2_8_14"]["label"] == "shortlist_full_64"
        assert decode_r8_routes[tok]["source_layer_20"]["label"] == "shortlist_source_1024"

    # Single-row decode always uses native
    d_r1 = run_dispatch(dense_plans, "decode", 1, 16384)
    s_r1 = run_dispatch(source_plans, "decode", 1, 32768)
    assert d_r1["label"] == "native_decode"
    assert s_r1["label"] == "native_decode"

    return {
        "opt_in_state_bytes": 100,
        "prefill_routes": prefill_routes,
        "decode_r8_routes": decode_r8_routes,
    }


def main():
    ast_res = check_ast_invariants()
    scope, model_scope = build_dispatch_harness()
    def_res = test_default_off(scope, model_scope)
    opt_res = test_opt_in_routing(scope, model_scope)
    report = {
        "passed": True,
        "ast_invariants": ast_res,
        "default_off": def_res,
        "opt_in_routing": opt_res,
    }
    out_path = EXP / "runs/local/cpu-routing-check.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps({"passed": True, "receipt": str(out_path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
