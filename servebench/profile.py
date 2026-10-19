"""Small fixed-shape workloads for attributing profiled engine costs."""

import json
import urllib.request
import uuid

from . import engine, prom


def one_row(base_url: str, *, requests: int = 32, prompt_token: int = 1) -> dict:
    """Profile cold single-token target forwards, without chat-template padding.

    A fresh cache salt per request ensures the token is computed, even with prefix
    caching. This is a cost probe, not a real-text throughput benchmark.
    """
    if requests < 24:
        raise ValueError("one-row profiling needs at least 24 engine iterations")
    base = base_url.rstrip("/")
    prom.require_idle(base, "one-row profiling")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]

    def forward():
        result = engine._post(base + "/v1/completions", {
            "model": model, "prompt": [prompt_token], "max_tokens": 1,
            "temperature": 0, "cache_salt": uuid.uuid4().hex,
        }, 120)
        if result["usage"]["prompt_tokens"] != 1:
            raise RuntimeError("one-row profile received a padded prompt")
        return result

    for _ in range(2):
        forward()
    engine._post(base + "/start_profile", None, 120)
    try:
        for _ in range(requests):
            forward()
    finally:
        engine._post(base + "/stop_profile", None, 120)
    return {"requests": requests, "prompt_tokens_per_request": 1}


def ced_moe_costs(events: list[dict], *, rows: int, decoder_layers: int = 20) -> list[dict]:
    """CED's routed FFN work, excluding the boundary's unchanged full-row KV work.

    For r6x TP3, 128 rows use 408 routed tiles; a single real row uses six.
    Fail on unfamiliar geometry or an incomplete decoder instead of guessing.
    One-row traces include the 40 target layers before draft preparation; select
    their decoder half. Nonfinal prefill traces contain complete decoder halves
    even when the profiler cuts off the preceding encoder's first layers.
    """
    if rows not in (1, 128):
        raise ValueError("CED attribution supports one or 128 rows")
    kernels = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
    if rows == 1:
        prepare = next((i for i, e in enumerate(kernels) if '_prepare_dflash_inputs' in e['name']), None)
        if prepare is None:
            raise ValueError('missing target/drafter boundary')
        kernels = kernels[:prepare]
    tiles = 408 if rows == 128 else 6
    routed = [e for e in kernels if e['name'] == 'moe_grouped_fp8_fp4_gemm_kernel'
              and e.get('args', {}).get('grid', [None, None])[1] == tiles]
    if rows == 1:
        if len(routed) != 4 * decoder_layers:
            raise ValueError(f'expected 40 target FFNs, found {len(routed) // 2}')
        routed = routed[2 * decoder_layers:]
    width = 2 * decoder_layers
    if not routed or len(routed) % width:
        raise ValueError('incomplete CED decoder FFN group')
    result = []
    for offset in range(0, len(routed), width):
        group = routed[offset:offset + width]
        if [e['args']['grid'][0] for e in group] != [12, 40] * decoder_layers:
            raise ValueError('unexpected routed projection order')
        result.append({'routed_ffn_ms': sum(e['dur'] for e in group) / 1000,
                       'decoder_body_span_ms': (group[-1]['ts'] + group[-1]['dur'] - group[0]['ts']) / 1000})
    return result
