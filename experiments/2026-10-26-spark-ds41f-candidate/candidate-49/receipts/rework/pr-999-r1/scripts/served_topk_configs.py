#!/usr/bin/env python3
"""Enumerate the IndexerTopK configurations serving declares, from the image's code.

For every attention layer of DeepSeek-V4.1-Flash-1017 that owns an indexer, this
derives the layer attributes exactly as DeepseekV4Attention.__init__ does
(vllm/models/deepseek_v4_1/attention.py:488-514, 465-474), then runs the image's
own DeepseekV4Attention._attention_geometry, DeepseekV41TileLangAttention._plan_args
and tilelang.attention._index_calls / _index_plan (the declared kernel calls; prepare
freezes the kernel cache, so an undeclared call raises KernelNotPreparedError) for
each configured serving shape. Nothing is compiled or launched.

usage: served_topk_configs.py <hf config.json> <out.json>
"""
import json
import os
import sys
from types import SimpleNamespace

from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.ced import ced_decoder_start
from vllm.models.deepseek_v4_1.tilelang import attention as tla
from vllm.models.deepseek_v4_1.tilelang import indexer as ix

hf_path, out_path = sys.argv[1], sys.argv[2]
text = json.load(open(hf_path))["text_config"]
hf = SimpleNamespace(**text)

BLOCK = 256  # serve_args --block-size 256 (releases/spark-ds41f-1107/cluster.json)
DRAFTS = 5  # cluster.json "drafts": num_speculative_tokens of the dspark config
# Shapes of releases/spark-ds41f-1107/cluster.json (switched shapes extend these).
SHAPES = {
    "tp4-ring (tp4..tp8-switched)": dict(max_model_len=1048576, max_num_seqs=16, mnbt=16384),
    "tp3-triangle (tp3-switched)": dict(max_model_len=1048576, max_num_seqs=8, mnbt=6144),
    "tp3-triangle-4k": dict(max_model_len=262144, max_num_seqs=8, mnbt=4096),
}

SEG = ix.indexer_topk_segment_v41
MERGE = ix.indexer_topk_merge_v41


def layer_stub(layer_id, shape):
    """The attributes DeepseekV4Attention.__init__ sets, for one layer and shape."""
    s = SimpleNamespace()
    s.layer_id = layer_id
    s.capacity = shape["mnbt"]  # :488
    s.max_model_len = shape["max_model_len"]  # :489
    s.compress_ratio = (
        hf.compress_ratios[layer_id] if layer_id < len(hf.compress_ratios) else 0
    )  # :490-494
    s._main_page = BLOCK // max(s.compress_ratio, 1)  # :495-497
    s._main_width = (s.max_model_len + BLOCK - 1) // BLOCK  # :498-500
    s._index_page, s._index_width = s._main_page, s._main_width  # :501
    s.is_index_source = layer_id in hf.index_source_layer_ids  # :503
    s.candidate_source_layer = hf.candidate_source_layer_id  # :514
    s.is_draft = False  # target layers: layer_id < num_hidden_layers (:466-470)
    boundary = ced_decoder_start(hf)
    s.is_ced_decoder = boundary is not None and layer_id >= boundary  # :471-474
    s.indexer = object() if s.is_index_source else None  # :616-635
    s.topk_indices_buffer = None
    s.swa_width = hf.sliding_window
    s.n_local_heads = 0  # MLA plan args are not inspected here
    s.swa_cache_layer = SimpleNamespace(block_size=BLOCK)
    s.rotary_emb = SimpleNamespace(cos_sin_cache=SimpleNamespace(device="cpu"))
    s.DECODE_CHUNK = DeepseekV4Attention.DECODE_CHUNK
    s.INDEX_CHUNK = DeepseekV4Attention.INDEX_CHUNK
    s.config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            parallel_drafting=False,
            num_speculative_tokens=DRAFTS,
            num_verification_tokens=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=shape["max_num_seqs"]),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
    )
    s._attention_geometry = lambda: DeepseekV4Attention._attention_geometry(s)
    return s


def topk_configs(calls):
    """IndexerTopK configurations behind a declared call list, from its segment kernels."""
    found = []
    for function, args, kwargs in calls:
        if function is SEG:
            width, topk, group, final, candidates = args[:5]
            gather = bool(args[6]) if len(args) > 6 else False
            found.append(
                dict(width=width, topk=topk, group=group, candidates=bool(candidates),
                     gather_scores=gather, final_segment=bool(final))
            )
    return found


def route(cfg):
    segs, _, levels = ix.IndexerTopK.routes(
        cfg["width"], cfg["topk"], cfg["group"], cfg["candidates"],
        gather_scores=cfg["gather_scores"],
    )
    return dict(segments=segs, merge_levels=[[lists, out] for _, lists, out in levels])


def key(call):
    function, args, kwargs = call
    return (function.__name__ if hasattr(function, "__name__") else repr(function), repr(args), repr(sorted(kwargs.items())))


slow_calls = ix.IndexerTopK.calls(1048576, 512, 1, False)
slow_keys = {key(c) for c in slow_calls}

report = {
    "lsa_env": os.environ.get("VLLM_DS41_LSA"),
    "hf": {k: text[k] for k in ("num_hidden_layers", "compress_ratios", "index_source_layer_ids",
                                 "kv_source_layer_ids", "candidate_source_layer_id", "index_topk",
                                 "max_position_embeddings")},
    "ced_decoder_start": ced_decoder_start(hf),
    "slow_selector": "IndexerTopK(1048576, 512, 1) (raw_1m_k512_g1)",
    "slow_selector_calls": [repr(c[1]) for c in slow_calls],
    "slow_selector_route": route(dict(width=1048576, topk=512, group=1, candidates=False, gather_scores=False)),
    "shapes": {},
}
all_declared = set()
for shape_name, shape in SHAPES.items():
    layers = {}
    for layer_id in range(hf.num_hidden_layers):
        s = layer_stub(layer_id, shape)
        if s.indexer is None:
            continue
        _, index, chunks = tla.DeepseekV41TileLangAttention._plan_args(s)
        modes = {}
        declared = []
        # _plan_calls: one _index_calls per distinct plan argument tuple (tilelang/attention.py:196-198)
        for args in dict.fromkeys(index.values()):
            declared.extend(tla._index_calls(*args))
        for mode, args in index.items():
            calls = tla._index_calls(*args)
            cfgs = topk_configs(calls)
            modes[mode] = dict(
                plan_args=dict(width=args[0], page=args[1], candidates=args[2], candidate_blocks=args[3]),
                rows_per_call=chunks[mode],
                plan=("BlockMeanShortlistPlan" if tla._shortlist_options(args[0], args[2], args[3]) is not None
                      else "IndexerV41Plan"),
                topk_configs=[dict(**c, route=route(c)) for c in cfgs],
            )
        dk = {key(c) for c in declared}
        all_declared |= dk
        layers[str(layer_id)] = dict(
            compress_ratio=s.compress_ratio,
            index_width=s._index_width,
            index_page=s._index_page,
            role=("source" if layer_id == s.candidate_source_layer else
                  "reindex" if layer_id > s.candidate_source_layer else "dense"),
            is_ced_decoder=s.is_ced_decoder,
            short_index_shape=list(s._short_index_shape) if s._short_index_shape else None,
            modes=modes,
            declares_slow_selector_kernels=sorted(k[1] for k in (dk & slow_keys)),
        )
    report["shapes"][shape_name] = dict(sizing=shape, layers=layers)

report["slow_selector_kernels_declared_anywhere"] = sorted(k[1] for k in (all_declared & slow_keys))
report["slow_selector_reachable"] = bool(all_declared & slow_keys)
uniq = {}
for shape_name, sh in report["shapes"].items():
    for lid, l in sh["layers"].items():
        for mode, m in l["modes"].items():
            for c in m["topk_configs"]:
                k = (c["width"], c["topk"], c["group"], c["candidates"], c["gather_scores"])
                uniq.setdefault(k, set()).add(f"{shape_name} L{lid} {mode} rows<={m['rows_per_call']}")
report["distinct_served_topk_configs"] = [
    dict(width=k[0], topk=k[1], group=k[2], candidates=k[3], gather_scores=k[4],
         route=route(dict(width=k[0], topk=k[1], group=k[2], candidates=k[3], gather_scores=k[4])),
         used_by=sorted(v))
    for k, v in sorted(uniq.items())
]
json.dump(report, open(out_path, "w"), indent=1)
print(json.dumps({k: report[k] for k in ("lsa_env", "ced_decoder_start", "slow_selector_route",
                                         "slow_selector_reachable", "slow_selector_kernels_declared_anywhere")}, indent=1))
for c in report["distinct_served_topk_configs"]:
    print(json.dumps({k: c[k] for k in ("width", "topk", "group", "candidates", "gather_scores", "route")}))
    for u in c["used_by"]:
        print("    ", u)
