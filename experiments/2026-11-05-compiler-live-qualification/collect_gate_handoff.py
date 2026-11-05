#!/usr/bin/env python3
"""Collect C42 image evidence only after gate-only normal closure and independent restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = "/home/swank/projects/spark-ds41f-copy-default"
OUT = ROOT + "/results/private/candidate-rounds/candidate-42-image"
IMAGE = "sha256:70e9abba12d83b79a268574a2057b096af2b9fc5fcbc1b07364d6eb580b1a35a"
PIN = "b6896b389c5bb0004e13fe9f4f941a31a60f87ca"


def ssh(host, program, **kwargs):
    compile(program, "remote-gate-export", "exec")
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
         "python3 -c " + shlex.quote(program)], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing an existing export")
    head_guard = "PIN=" + repr(PIN) + "\n" + r'''
import hashlib,json,pathlib
h=pathlib.Path.home();root=h/'projects/spark-ds41f';setup=h/'tl-logs/compiler-live-qualification/setup'
rp=root/'results/private/lab/candidate-42-round/receipt.json';ep=setup/'exit.json'
r=json.loads(rp.read_text()) if rp.exists() else {};e=json.loads(ep.read_text()) if ep.exists() else {}
ready=not r.get('completed') and r.get('ended') and r.get('restored') and r.get('error')=='RuntimeError: window changed or closure requested' and not r.get('restore_error') and r.get('commit')==PIN and e.get('actual_wait_exit')==1 and [(v.get('log'),v.get('exit')) for v in r.get('steps',[])]==[('image-and-gate.log',0)]
if not ready:
 print(json.dumps({'ready':False,'reason':'owned gate-only closure not ended/restored with actual wait exit1 and no serving phase','phase':r.get('phase')}));raise SystemExit(0)
shared={}
for key,name in [('hold','spark-hold.json'),('request','spark-request.json'),('legacy','spark3-request.json')]:
 p=h/name;shared[key]=json.loads(p.read_text()) if p.exists() else None
if shared['hold'] or shared['legacy']:
 print(json.dumps({'ready':False,'reason':'new shared window or legacy request; defer builder export','shared':shared}));raise SystemExit(0)
dp=setup/'gate-handoff-restore-doctor.json';np=setup/'gate-handoff-restoration-ranks.json'
if not dp.exists() or not np.exists():
 print(json.dumps({'ready':False,'reason':'independent restoration receipts missing'}));raise SystemExit(0)
assert json.loads(dp.read_text())['exit']==0
rows=json.loads(np.read_text());assert rows['passed'] and {v['host'] for v in rows['ranks']}=={'dgx1','dgx2','dgx3'}
qualified=root/'results/private/candidate-rounds/candidate-42-qualified-image/receipt.json'
print(json.dumps({'ready':True,'receipt_sha256':hashlib.sha256(qualified.read_bytes()).hexdigest(),'shared':shared}))
'''
    p = ssh("dgx1", head_guard, capture_output=True, text=True, timeout=25)
    if p.returncode:
        raise SystemExit(p.stderr)
    gate = json.loads(p.stdout)
    if not gate["ready"]:
        print(json.dumps({"export_deferred_before_builder_reads": True, **gate}))
        return 2
    receipt_sha = gate["receipt_sha256"]
    program = "OUT=" + repr(OUT) + "\nIMAGE=" + repr(IMAGE) + "\nEXPECTED=" + repr(receipt_sha) + "\n" + r'''
import datetime,hashlib,json,pathlib,subprocess
out=pathlib.Path(OUT)
data=(out/'receipt.json').read_bytes()
assert hashlib.sha256(data).hexdigest()==EXPECTED
r=json.loads(data)
assert r.get('passed') is True and r.get('ended') and not r.get('error') and r['image']==IMAGE
assert len(r['steps'])==9 and all(s['exit']==0 for s in r['steps'])
assert not r['conditions']['halted'] and not r['conditions']['fan_stop_failures']
image=json.loads(subprocess.check_output(['docker','image','inspect','spark-ds41f:candidate-42'],text=True))[0]
assert image['Id']==IMAGE
labels=image['Config'].get('Labels') or {}
assert all(labels.get(k)==v for k,v in r['source']['expected_labels'].items())
assert labels.get('local.spark-ds41f.vllm.patch-head')==r['source']['vllm_head']
cache_names={'tilelang-cache','cuda-cache','torch-extensions'}
# Keep compiled caches node-private and untouched; export all portable original
# receipts, sources, logs and XML outside these explicitly declared cache roots.
paths=[p for p in sorted(out.rglob('*')) if p.is_file() and not (set(p.relative_to(out).parts)&cache_names)]
assert all(not p.is_symlink() for p in paths)
assert sum(p.stat().st_size for p in paths)<128*1024*1024
files={}
for p in paths:
 b=p.read_bytes(); files[str(p.relative_to(out))]={'source':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'dgx4','image':IMAGE,'actual_image_labels':{k:v for k,v in labels.items() if k.startswith('local.spark-ds41f.')},'files':files,'node_private_cache_subtrees_retained_unmodified':sorted(cache_names)}))
'''
    p = ssh("dgx4", program, capture_output=True, text=True, timeout=40)
    if p.returncode:
        raise SystemExit(p.stderr)
    receipt = json.loads(p.stdout)
    args.output.mkdir(parents=True)
    (args.output / ".gitignore").write_text("raw/\nraw.tar\n")
    program = "files=" + repr(receipt["files"]) + "\nOUT=" + repr(OUT) + "\nEXPECTED=" + repr(receipt_sha) + "\n" + r'''
import hashlib,pathlib,sys,tarfile
assert hashlib.sha256((pathlib.Path(OUT)/'receipt.json').read_bytes()).hexdigest()==EXPECTED
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,record in sorted(files.items()):
  archive.add(record['source'],arcname=name,recursive=False)
'''
    archive = args.output / "raw.tar"
    with archive.open("xb") as handle:
        p = ssh("dgx4", program, stdout=handle, stderr=subprocess.PIPE, timeout=90)
    if p.returncode:
        raise SystemExit(p.stderr.decode())
    raw = args.output / "raw"
    raw.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(raw, filter="data")
    assert {str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()} == set(receipt["files"])
    for name, record in receipt["files"].items():
        data = (raw / name).read_bytes()
        assert len(data) == record["size"] and hashlib.sha256(data).hexdigest() == record["sha256"], name
    receipt.update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   all_native_hashes_verified_locally=True)
    (args.output / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "verified_files": len(receipt["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
