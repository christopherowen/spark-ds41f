#!/usr/bin/env python3
"""Retain C41's completed image gate after the queued foreign window releases."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = "/home/swank/projects/spark-ds41f-copy-default"
OUT = ROOT + "/results/private/candidate-rounds/candidate-41-image"
IMAGE = "sha256:38ff08bb1eb2495cb934e008b6f31a482aaa238c08b959aaf72cf2cb5658d29c"
RECEIPT_SHA = "00b15d2c51020ebf8c806ba29f6b962199cf2d4bb028b010686b9c655ce1f6b7"


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
    p = ssh("dgx1", "import json,pathlib\nr={}\n"
            "for key,name in [('hold','spark-hold.json'),('request','spark-request.json'),('legacy','spark3-request.json')]:\n"
            " p=pathlib.Path.home()/name; r[key]=json.loads(p.read_text()) if p.exists() else None\n"
            "print(json.dumps(r))",
            capture_output=True, text=True, timeout=20)
    if p.returncode:
        raise SystemExit(p.stderr)
    shared = json.loads(p.stdout)
    request = shared.get("request") or {}
    if (shared["hold"] or shared["legacy"]
            or request.get("who") != "copy-default-screening"
            or request.get("source_commit") != "0da35cc9668ff491065446b2a842fe996a73e80f"
            or request.get("since") != "2026-11-05T07:22:10.331161+00:00"):
        print(json.dumps({"export_deferred_before_builder_reads": True, "shared": shared}))
        return 2
    program = "OUT=" + repr(OUT) + "\nIMAGE=" + repr(IMAGE) + "\nEXPECTED=" + repr(RECEIPT_SHA) + "\n" + r'''
import datetime,hashlib,json,pathlib,subprocess
out=pathlib.Path(OUT)
data=(out/'receipt.json').read_bytes()
assert hashlib.sha256(data).hexdigest()==EXPECTED
r=json.loads(data)
assert r.get('passed') is True and r.get('ended') and not r.get('error') and r['image']==IMAGE
assert len(r['steps'])==8 and all(s['exit']==0 for s in r['steps'])
assert not r['conditions']['halted'] and not r['conditions']['fan_stop_failures']
image=json.loads(subprocess.check_output(['docker','image','inspect','spark-ds41f:candidate-41'],text=True))[0]
assert image['Id']==IMAGE
labels=image['Config'].get('Labels') or {}
assert all(labels.get(k)==v for k,v in r['source']['expected_labels'].items())
assert labels.get('local.spark-ds41f.vllm.patch-head')==r['source']['vllm_head']
paths=[p for p in sorted(out.rglob('*')) if p.is_file()]
assert all(not p.is_symlink() for p in paths)
assert sum(p.stat().st_size for p in paths)<128*1024*1024
files={}
for p in paths:
 b=p.read_bytes(); files[str(p.relative_to(out))]={'source':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'dgx4','image':IMAGE,'actual_image_labels':{k:v for k,v in labels.items() if k.startswith('local.spark-ds41f.')},'files':files}))
'''
    p = ssh("dgx4", program, capture_output=True, text=True, timeout=40)
    if p.returncode:
        raise SystemExit(p.stderr)
    receipt = json.loads(p.stdout)
    args.output.mkdir(parents=True)
    (args.output / ".gitignore").write_text("raw/\nraw.tar\n")
    program = "files=" + repr(receipt["files"]) + "\nOUT=" + repr(OUT) + "\nEXPECTED=" + repr(RECEIPT_SHA) + "\n" + r'''
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
