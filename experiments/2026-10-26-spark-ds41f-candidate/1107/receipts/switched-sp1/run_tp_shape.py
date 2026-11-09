#!/usr/bin/env python3
import hashlib, json, pathlib, subprocess, sys, time

ROOT = pathlib.Path("/home/chris/projects/spark-ds41f")
LAB = pathlib.Path("/home/chris/spark-lab/1107-switched")
ALL_NODES = [f"sp{i}" for i in range(1, 9)]

def run_cmd(cmd, check=True):
    print(f"==> Running: {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=str(ROOT), check=check)

def sample_host(node):
    script = """
import json, pathlib, subprocess
rdma_names = ("roce_adp_retrans", "packet_seq_err", "out_of_sequence", "np_cnp_sent")
rdma = {}
for dev in sorted(pathlib.Path("/sys/class/infiniband").iterdir()):
    rdma[dev.name] = {n: int((dev / "ports/1/hw_counters" / n).read_text()) for n in rdma_names}
port_names = ("tx_bytes_phy", "rx_bytes_phy", "rx_out_of_buffer", "rx_discards_phy", "tx_discards_phy", "tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_bytes", "rx_prio0_Storm_warning_events", "rx_prio3_Storm_warning_events")
ports = {}
for hca in sorted(pathlib.Path("/sys/class/infiniband").iterdir()):
    for netdev in sorted((hca / "device/net").iterdir()):
        out = subprocess.check_output(["ethtool", "-S", netdev.name], text=True, timeout=10)
        vals = {k.strip(): int(v.strip()) for line in out.splitlines() if ":" in line for k, v in [line.split(":", 1)] if v.strip().isdigit()}
        ports[netdev.name] = {n: vals.get(n, 0) for n in port_names}
print(json.dumps({"rdma": rdma, "ports": ports}))
"""
    res = subprocess.check_output(["ssh", node, "python3", "-"], input=script, text=True)
    return json.loads(res)

def run_probe(tp_num):
    cfg = f"experiments/2026-10-26-spark-ds41f-candidate/1107/tp{tp_num}-switched.json"
    nodes = [f"sp{i}" for i in range(1, tp_num + 1)]
    outdir = LAB / f"tp{tp_num}"
    outdir.mkdir(parents=True, exist_ok=True)

    for n in ALL_NODES:
        subprocess.run(["ssh", n, "docker rm -f spark-ds41f spark-collective-probe >/dev/null 2>&1 || true; pkill -f memguard.sh || true; rm -rf /dev/shm/psm_* /dev/shm/sparknet_* 2>/dev/null || true"], check=False)

    before = {n: sample_host(n) for n in nodes}
    cmds = {}
    for n in nodes:
        out = subprocess.check_output(["bin/spark", "--cluster-config", cfg, "topology", "probe", n], cwd=str(ROOT), text=True)
        cmd = [line for line in out.splitlines() if line.startswith("docker run")][0]
        cmds[n] = cmd + " --benchmark --counter-samples --numerics"

    procs = {}
    for n in nodes:
        logf = open(outdir / f"probe-collectives-{n}.log", "w")
        procs[n] = (subprocess.Popen(["ssh", n, cmds[n]], stdout=logf, stderr=subprocess.STDOUT, text=True), logf)

    rc = {}
    for n, (p, logf) in procs.items():
        rc[n] = p.wait()
        logf.close()

    after = {n: sample_host(n) for n in nodes}
    rdma_deltas = {}
    port_deltas = {}
    for n in nodes:
        rdma_deltas[n] = {dev: {k: after[n]["rdma"][dev][k] - before[n]["rdma"][dev][k] for k in before[n]["rdma"][dev]} for dev in before[n]["rdma"]}
        port_deltas[n] = {dev: {k: after[n]["ports"][dev][k] - before[n]["ports"][dev][k] for k in before[n]["ports"][dev]} for dev in before[n]["ports"]}

    ranks = {}
    for n in nodes:
        lines = [l for l in (outdir / f"probe-collectives-{n}.log").read_text().splitlines() if l.strip().startswith("{") and "world_size" in l]
        ranks[n] = json.loads(lines[-1][lines[-1].index("{"):]) if lines else {"passed": False, "rc": rc[n]}

    out_of_order = sum(d["out_of_sequence"] for n in nodes for d in rdma_deltas[n].values())
    packet_seq_err = sum(d["packet_seq_err"] for n in nodes for d in rdma_deltas[n].values())
    roce_adp_retrans = sum(d["roce_adp_retrans"] for n in nodes for d in rdma_deltas[n].values())
    np_cnp_sent = sum(d["np_cnp_sent"] for n in nodes for d in rdma_deltas[n].values())
    discard_packets = sum(d["rx_out_of_buffer"] + d["rx_discards_phy"] + d["tx_discards_phy"] for n in nodes for d in port_deltas[n].values())
    pfc_storms = sum(d["rx_prio0_Storm_warning_events"] + d["rx_prio3_Storm_warning_events"] for n in nodes for d in port_deltas[n].values())
    passed = all(rc[n] == 0 and ranks[n].get("passed", False) for n in nodes) and out_of_order == 0 and packet_seq_err == 0 and discard_packets == 0 and pfc_storms == 0

    summary = {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "out_of_order": out_of_order,
        "packet_seq_err": packet_seq_err,
        "roce_adp_retrans": roce_adp_retrans,
        "np_cnp_sent": np_cnp_sent,
        "discard_packets": discard_packets,
        "pfc_storms": pfc_storms,
        "rdma_error_deltas": rdma_deltas,
        "port_deltas": port_deltas,
        "rank0": ranks["sp1"],
        "ranks": ranks,
    }
    (outdir / "probe-collectives.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"TP{tp_num} PROBE:", json.dumps({k: summary[k] for k in ("status", "passed", "out_of_order", "packet_seq_err", "roce_adp_retrans", "np_cnp_sent", "discard_packets", "pfc_storms")}), flush=True)
    if not passed:
        raise RuntimeError(f"TP{tp_num} probe_collectives failed!")

def record_mem(tp_num, boot):
    nodes = [f"sp{i}" for i in range(1, tp_num + 1)]
    outdir = LAB / f"tp{tp_num}"
    mem = {}
    for n in nodes:
        out = subprocess.check_output(["ssh", n, "grep MemAvailable /proc/meminfo"], text=True).strip()
        kb = int(out.split()[1])
        mem[n] = round(kb / (1024 * 1024), 2)
    print(f"TP{tp_num} Boot {boot} MemAvailable GiB:", mem, flush=True)
    (outdir / f"boot{boot}-mem.json").write_text(json.dumps(mem, indent=2) + "\n")

def run_shape(tp_num):
    cfg = f"experiments/2026-10-26-spark-ds41f-candidate/1107/tp{tp_num}-switched.json"
    ref = f"manifests/benchmarks/2026-10-30-spark-ds41f-1030-switched{tp_num}.json"
    outdir = LAB / f"tp{tp_num}"
    outdir.mkdir(parents=True, exist_ok=True)

    run_probe(tp_num)

    # Boot 1
    run_cmd(f"bin/spark --cluster-config {cfg} cluster start --replace --change-shape")
    run_cmd(f"bin/spark --cluster-config {cfg} cluster start --replace --change-shape --apply")
    record_mem(tp_num, 1)
    run_cmd(f"bin/spark --cluster-config {cfg} bench --suites quality,decode --compare {ref} --output {outdir}/spark-bench-boot1", check=False)
    if not (outdir / "spark-bench-boot1/bench.json").is_file():
        raise RuntimeError(f"TP{tp_num} Boot 1 bench.json not written!")
    run_cmd(f"cp {outdir}/spark-bench-boot1/bench.json {outdir}/spark-bench-boot1.json")

    # Boot 2
    run_cmd(f"bin/spark --cluster-config {cfg} cluster start --replace")
    run_cmd(f"bin/spark --cluster-config {cfg} cluster start --replace --apply")
    record_mem(tp_num, 2)
    run_cmd(f"bin/spark --cluster-config {cfg} bench --suites quality,decode --compare {ref} --output {outdir}/spark-bench-boot2", check=False)
    if not (outdir / "spark-bench-boot2/bench.json").is_file():
        raise RuntimeError(f"TP{tp_num} Boot 2 bench.json not written!")
    run_cmd(f"cp {outdir}/spark-bench-boot2/bench.json {outdir}/spark-bench-boot2.json")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run_shape(int(arg))
