"""Fabric script: the GID slot each of this rank's HCAs was found in, against the sysfs table.

The lab runs it on every node at once in the arm's image (``fabric`` job) with
the probe's rank arguments. It builds sparknet's one-shot runtime from the
arm's environment (``SPARKNET_ROCE_SUBNETS`` included), then for every opened
HCA compares the slot the proxy's ``ibv_query_gid_ex`` scan took
(``stats()["gid_indices"]``) with sparknet's rule applied to the port's sysfs
GID table, and checks one all-reduce over the result. It prints the rank's
report and exits 0 when the two agree and the sum is right.
"""

import argparse
import json
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from sparknet import oneshot
from sparknet.topology.gids import GidSelectionError, read_gid_table, select_gid


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for option in ("--rank", "--world-size", "--master-addr", "--master-port"):
        parser.add_argument(option, required=True)
    args, _ = parser.parse_known_args(argv)
    rank, world = int(args.rank), int(args.world_size)
    dist.init_process_group("gloo", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            rank=rank, world_size=world, timeout=timedelta(seconds=120))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    runtime = oneshot.AllReduce.from_exchange_group(exchange_group=dist.group.WORLD, device=device)
    runtime.prepare((torch.bfloat16,))
    stats = runtime.stats()
    report = {"rank": rank, "gid_indices": stats["gid_indices"], "gid_subnets": stats["gid_subnets"],
              "sysfs": {}, "problems": []}
    for hca in runtime.hca_names:
        table = read_gid_table(f"/sys/class/infiniband/{hca}/ports/1")
        report["sysfs"][hca] = table
        try:
            index, _ = select_gid(table, stats["gid_subnets"][hca], hca)
        except GidSelectionError as error:
            report["problems"].append(f"sysfs: {error}")
            continue
        if index != stats["gid_indices"][hca]:
            report["problems"].append(f"{hca}: the proxy took slot {stats['gid_indices'][hca]}, sysfs gives {index}")
    value = torch.full((4096,), rank + 1, dtype=torch.bfloat16, device=device)
    total = runtime.all_reduce(value)
    torch.cuda.synchronize()
    if not torch.equal(total, torch.full_like(total, world * (world + 1) // 2)):
        report["problems"].append("the all-reduce sum is wrong")
    runtime.close()
    report["passed"] = not report["problems"]
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
