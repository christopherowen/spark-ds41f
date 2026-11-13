"""Extend 0778's attention test with the decode step of an SP forward (run in the fix worktree)."""

from pathlib import Path

p = Path("tests/models/test_deepseek_v4_1_attention.py")
s = p.read_text()

edits = [
    (
        '''@pytest.mark.parametrize("n_real", [4, 2, 0])
@pytest.mark.parametrize("split_check", [False, True])
def test_sp_prefill_with_replicated_fused_wqa_wkv_gathers_mxfp8_and_swa_records(
    monkeypatch, n_real, split_check
):
    """When ``fused_wqa_wkv`` is replicated across TP, SP prefill without a compressor
    runs ``sp_attention_front`` on ``qr_kv[:real_rows]``, gathers only ``local_q_cast``
    and 528-byte SWA cache records via ``gather_mxfp8`` (1,848 B/row), scatters the
    records via ``write_swa_records``, and keeps ``indexer.weights_proj`` rank-local
    unless ``_INDEX_SPLIT_CHECK`` is active."""
''',
        '''@pytest.mark.parametrize("is_decode", [False, True])
@pytest.mark.parametrize("n_real", [4, 2, 0])
@pytest.mark.parametrize("split_check", [False, True])
def test_sp_prefill_with_replicated_fused_wqa_wkv_gathers_mxfp8_and_swa_records(
    monkeypatch, n_real, split_check, is_decode
):
    """When ``fused_wqa_wkv`` is replicated across TP, SP prefill without a compressor
    runs ``sp_attention_front`` on ``qr_kv[:real_rows]``, gathers only ``local_q_cast``
    and 528-byte SWA cache records via ``gather_mxfp8`` (1,848 B/row), scatters the
    records via ``write_swa_records``, and keeps ``indexer.weights_proj`` rank-local
    unless ``_INDEX_SPLIT_CHECK`` is active. A decode step of an SP forward (the
    warm-up's eager tail shapes) scores every row, so it gathers the head weights of
    every row: the index queries and their weights always have one row count."""
''',
    ),
    (
        '''    metadata = {"swa": SimpleNamespace(is_decode=False, slot_mapping=slots)}
''',
        '''    metadata = {"swa": SimpleNamespace(is_decode=is_decode, slot_mapping=slots)}
''',
    ),
    (
        '''    if split_check:
        expected_events.append(("gather", (4, 4), torch.bfloat16))
    if n_real > 0:
''',
        '''    if split_check or is_decode:
        expected_events.append(("gather", (4, 4), torch.bfloat16))
    if n_real > 0:
''',
    ),
    (
        '''    iq_data, iq_scale, weights, split, ref = seen_index_query[0]
    assert iq_data.shape == (n_real, 4, 64)
    assert iq_scale.shape == (n_real, 4, 4)
    if split_check:
''',
        '''    iq_data, iq_scale, weights, split, ref = seen_index_query[0]
    assert weights.shape[0] == iq_data.shape[0] == iq_scale.shape[0]
    if is_decode:
        assert split is None and ref is None
        assert iq_data.shape == (10, 4, 64)
        assert iq_scale.shape == (10, 4, 4)
        assert torch.equal(weights, gathered_weights)
        return
    assert iq_data.shape == (n_real, 4, 64)
    assert iq_scale.shape == (n_real, 4, 4)
    if split_check:
''',
    ),
]
for old, new in edits:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
p.write_text(s)
print("test extended")
