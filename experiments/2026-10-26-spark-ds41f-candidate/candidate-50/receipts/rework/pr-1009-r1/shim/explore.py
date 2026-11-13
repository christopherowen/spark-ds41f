"""Exploration: build the serving config and layers 20 and 24's attention, print specs."""

import sys
import types

import torch

sys.path.insert(0, "/shim")
import attn_harness as H  # noqa: E402

vc = H.make_config(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
hf = vc.model_config.hf_config
print("hf", type(hf).__name__, hf.num_attention_heads, hf.o_groups, hf.index_n_heads)
H.init_dist(vc)
mods = H.build_attention(vc)
for m in mods.values():
    print(m.prefix, "heads", m.n_local_heads, "groups", m.n_local_groups, "ratio", m.compress_ratio,
          "kv_src", m.is_kv_source, "idx_src", m.is_index_source, "comp", m.compressor is not None,
          "idx", m.indexer is not None, "fused tp", getattr(m.fused_wqa_wkv, "tp_size", None),
          "repl", getattr(m.fused_wqa_wkv, "replicated_sp_prefill", None),
          "main_page", m._main_page, "main_width", m._main_width, "swa_bs", m.swa_cache_layer.block_size,
          "decode_cap", getattr(m, "_decode_capacity", None))
    for c in m._owned_caches():
        spec = c.get_kv_cache_spec(vc)
        print("  cache", c.prefix if hasattr(c, "prefix") else "?", type(spec).__name__, spec.block_size,
              getattr(spec, "page_size_bytes", None), getattr(spec, "tokens_per_state", None))
    if m.compressor is not None:
        print("  compressor", type(m.compressor).__name__, m.compressor.compress_ratio,
              getattr(m.compressor, "state_cache", None))
