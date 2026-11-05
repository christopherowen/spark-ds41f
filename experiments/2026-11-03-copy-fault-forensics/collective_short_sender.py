#!/usr/bin/env python3
"""Run the pinned sparknet receive/reduction kernel against a short peer payload.

This is an isolated single-GPU protocol fixture, not a distributed serving repro.
It constructs the memory/flags left by completed payload delivery; there is no
NIC, proxy thread, process group, model or serving operation. Current inputs are
finite. Old receive-slot contents are deliberately specified controls, not
claimed observations from a production crash. The original kernel is unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path


SPARKNET_REVISION = "1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf"
SOURCE_HASHES = {
    "_oneshot_tilelang.py": "33e147fefb052d7e7f6ec1a705b99f550e80afd247293758c820aeca19a7fbde",
    "_device.py": "fcb626dd8a202a708a6bba3eb0e828a89eb6681d26bfd3fa68f9bc073e461eb7",
    "_kernels.py": "c37f6e60cd0bd1f6f2b58d408a996c142354685ab7f429b592ea2ef8a5041ac5",
    "_proxy.py": "7468fe4b3d8b9ed83615b4c3bbd3e0eb2e1c07de8244bfd0c8878595f0fb1968",
    "_roce_proxy.c": "256c711c72bc15fd51ed5177edb777742d2123398221e64790d5cfe5a34083f7",
}
CASES = {
    "equal-finite": (17, "finite-bf16"),
    "short-finite": (6, "finite-bf16"),
    "equal-int-tail": (17, "int64-minus-one"),
    "short-int-tail": (6, "int64-minus-one"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    args = parser.parse_args()

    import torch
    from sparknet.oneshot import _oneshot_tilelang
    from sparknet.oneshot._kernels import Launch
    from sparknet.oneshot._proxy import Layout

    source_root = Path(_oneshot_tilelang.__file__).resolve().parent
    observed = {}
    for name, expected in SOURCE_HASHES.items():
        observed[name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if observed[name] != expected:
            raise RuntimeError(f"Not pinned sparknet source: {name}: {observed[name]}")

    rows, width, world, rank, lanes = 17, 5120, 3, 0, 2
    peer_rows, old_kind = CASES[args.case]
    nbytes = rows * width * 2
    slot_bytes = ((nbytes + 4095) // 4096) * 4096
    layout = Layout(world, slot_bytes)
    if layout.slots != 2:
        raise RuntimeError(f"Unexpected slot count: {layout.slots}")
    region = torch.zeros(layout.total_bytes, dtype=torch.uint8, pin_memory=True)
    # The kernel's system-scope host-pointer path requires this GPU architecture.
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"Fixture is qualified only for the GB10 host-pointer path: {capability}")

    seq, slot = 1, 1
    for peer, fill in ((1, 2.0), (2, 3.0)):
        offset = layout.recv_off + (peer * layout.slots + slot) * slot_bytes
        receive = region[offset:offset + slot_bytes]
        if peer == 1 and old_kind == "int64-minus-one":
            # A legitimate previous integer payload has a different meaning
            # when an unwritten suffix is later consumed as BF16.
            receive.view(torch.int64).fill_(-1)
        else:
            receive.view(torch.bfloat16).fill_(5.0)
        written_rows = peer_rows if peer == 1 else rows
        receive[:written_rows * width * 2].view(torch.bfloat16).fill_(fill)
        for lane in range(lanes):
            flag = layout.flag_off + ((peer * layout.slots + slot) * lanes + lane) * layout.flag_stride
            region[flag:flag + 4].view(torch.int32).fill_(seq)
    # One-second device wait limit; all required peer flags are already present.
    ctrl = region[layout.ctrl_off:layout.ctrl_off + 40].view(torch.int32)
    ctrl[7] = 1_000_000
    input_tensor = torch.ones((rows, width), dtype=torch.bfloat16, device="cuda")
    output_tensor = torch.empty_like(input_tensor)
    counters = torch.zeros(5, dtype=torch.int32, device="cuda")
    launch = Launch(
        input=input_tensor.view(torch.int32).reshape(-1),
        output=output_tensor.view(torch.int32).reshape(-1),
        size_packs=nbytes // 16, nbytes=nbytes, region=region,
        recv_off=layout.recv_off, flag_off=layout.flag_off,
        send_off=layout.send_off, ctrl_off=layout.ctrl_off, slot_bytes=slot_bytes,
        counters=counters, stage_index=1, tail_index=2,
        rung_index=3, arrival_index=4, grid_x=1,
    )
    kernel = _oneshot_tilelang.get_launcher(
        "bfloat16", world, rank, 128, layout.slots, layout.flag_stride,
        lanes, torch.cuda.current_device(),
    )
    torch.cuda.nvtx.range_push("short-sender/" + args.case)
    kernel(launch)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    result = output_tensor.cpu().float()
    finite = torch.isfinite(result)
    bad_rows = torch.where(~finite.all(dim=1))[0].tolist()
    expected = torch.full_like(result, 6.0)
    if peer_rows < rows:
        expected[peer_rows:] = float("nan") if old_kind == "int64-minus-one" else 9.0
    matches = torch.equal(torch.isnan(result), torch.isnan(expected)) and torch.equal(
        result[torch.isfinite(expected)], expected[torch.isfinite(expected)]
    )
    state = counters.cpu().tolist()
    passed = matches and state[0] == seq and int(ctrl[2]) == 0
    print("SHORT-SENDER-RESULT " + json.dumps({
        "case": args.case, "passed": passed,
        "sparknet_revision": SPARKNET_REVISION,
        "source_root": str(source_root), "source_sha256": observed,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "capability": capability, "receiver_rows": rows, "peer_rows": [rows, peer_rows, rows],
        "row_width": width, "receiver_bytes": nbytes,
        "short_peer_written_bytes": peer_rows * width * 2,
        "old_receive_slot_kind": old_kind, "nonfinite_rows": bad_rows,
        "first_column": [None if not torch.isfinite(v) else float(v) for v in result[:, 0]],
        "device_counters": state, "control_words": ctrl.tolist(),
        "scope": "Real pinned reduction kernel; completed-delivery snapshot simulated, no NIC or model. Old tail is a specified control, not captured production bytes.",
    }, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
