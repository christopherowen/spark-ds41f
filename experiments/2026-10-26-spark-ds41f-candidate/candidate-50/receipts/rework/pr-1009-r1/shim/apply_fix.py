"""Apply the round 1 correction to 0778's attention.py (run in the fix worktree)."""

from pathlib import Path

p = Path("vllm/models/deepseek_v4_1/attention.py")
s = p.read_text()
old = '''        before one grouped ``gather_mxfp8`` (1,848 B/row) and ``write_swa_records``."""
        index_weights = None
        if self.indexer is not None:
            index_weights = self.indexer.weights_proj(hidden_states)
            if _INDEX_SPLIT_CHECK:
                index_weights = sp_rows.gather(index_weights)
            else:
                index_weights = index_weights[: sp_rows.real_rows]
        metadata = get_forward_context().attn_metadata
        swa_cache = getattr(self.swa_cache_layer, "kv_cache", None)
        if not is_breakable_cudagraph_enabled() and isinstance(metadata, dict):
            swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
            if (
'''
new = '''        before one grouped ``gather_mxfp8`` (1,848 B/row) and ``write_swa_records``.

        The head weights stay local only where ``_forward`` splits the indexer by
        rank: a step with prefill rows. A decode step, every request within the
        decode threshold (the warm-up's eager token counts past the capture limit,
        or a step of short requests), scores every row, as does the split check's
        reference, so there the head weights of every row are gathered. The index
        queries and their head weights then reach the scorer with one row count."""
        metadata = get_forward_context().attn_metadata
        swa = (
            self._query_metadata(metadata[self.swa_cache_layer.prefix])
            if isinstance(metadata, dict)
            else None
        )
        index_weights = None
        if self.indexer is not None:
            index_weights = self.indexer.weights_proj(hidden_states)
            if _INDEX_SPLIT_CHECK or (swa is not None and swa.is_decode):
                index_weights = sp_rows.gather(index_weights)
            else:
                index_weights = index_weights[: sp_rows.real_rows]
        swa_cache = getattr(self.swa_cache_layer, "kv_cache", None)
        if not is_breakable_cudagraph_enabled() and swa is not None:
            if (
'''
assert s.count(old) == 1, "0778 _sp_front text not found"
s = s.replace(old, new)
p.write_text(s)
print("attention.py corrected")
