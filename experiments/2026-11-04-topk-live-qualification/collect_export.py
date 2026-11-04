#!/usr/bin/env python3
"""Download immutable C40 outputs only after actual exit and independent restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = "/home/swank/projects/spark-ds41f"
SETUP = "/home/swank/tl-logs/topk-live-qualification/setup"
PIN = "1f134c7aad82bc63c95afed36f488becf2959a41"

PROBE = r'''
import hashlib,json,pathlib,datetime
root=pathlib.Path(ROOT)
setup=pathlib.Path(SETUP)
control=root/'results/private/lab/candidate-40-round-attempt-2/receipt.json'
exit_path=setup/'exit-attempt-2.json'
r=json.loads(control.read_text())
e=json.loads(exit_path.read_text()) if exit_path.exists() else None
if not (r.get('completed') and r.get('restored') and r.get('ended') and
        not r.get('error') and not r.get('restore_error') and
        r.get('commit')==PIN and e and e.get('actual_exit')==0):
 print(json.dumps({'ready':False,'reason':'controller not successfully completed and restored with actual wait exit0','phase':r.get('phase'),'ended':r.get('ended')}))
 raise SystemExit(0)
doctor=setup/'post-round-restore-doctor.json'
ranks=setup/'post-round-restoration-ranks.json'
if not doctor.exists() or not ranks.exists():
 print(json.dumps({'ready':False,'reason':'independent restoration receipts missing'}))
 raise SystemExit(0)
assert json.loads(doctor.read_text())['exit']==0
rows=json.loads(ranks.read_text())['ranks']
assert {x['host'] for x in rows}=={'dgx1','dgx2','dgx3'}
assert all(x['running'] is True and x['image']=='sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11' and x['profile']=='experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json' for x in rows)
directories=[
 'results/private/lab/candidate-40-round-attempt-2',
 'results/private/determinism/candidate-40-functional-tp3',
 'results/private/determinism/candidate-40-release-tp3',
 'results/private/suites/candidate-40-release-tp3',
 'results/private/suites/candidate-40-release-tp3-single-stream',
 'results/private/suites/candidate-40-release-tp3-sampled',
]
paths={}
for name in directories:
 directory=root/name
 assert directory.is_dir(),name
 for p in sorted(directory.rglob('*')):
  if p.is_file():
   assert not p.is_symlink(),str(p)
   paths[str(p.relative_to(root))]=p
for pattern in ['candidate-40*table.txt','candidate-40-functional-tp3-trace-*.json']:
 for p in sorted((root/'results/private/lab').glob(pattern)):
  assert p.is_file() and not p.is_symlink()
  paths[str(p.relative_to(root))]=p
qualified=root/'results/private/candidate-rounds/candidate-40-qualified-image/receipt.json'
paths[str(qualified.relative_to(root))]=qualified
paths.update({
 'control/exit.json':exit_path,
 'control/launch.json':setup/'launch-attempt-2.json',
 'control/controller.log':setup.parent/'candidate-40-window-attempt-2.log',
 'control/restore-live-doctor.json':doctor,
 'control/restoration-ranks.json':ranks,
})
files={}
for name,p in paths.items():
 assert p.is_file() and not p.is_symlink(),str(p)
 data=p.read_bytes()
 files[name]={'source':str(p),'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps({'ready':True,'host':'dgx1','captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_commit':PIN,'files':files}))
'''


def ssh(program, **kwargs):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1",
         "python3 -c " + shlex.quote(program)], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing export output")
    probe = "ROOT=" + repr(ROOT) + "\nSETUP=" + repr(SETUP) + "\nPIN=" + repr(PIN) + "\n" + PROBE
    p = ssh(probe, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise SystemExit(p.stderr)
    receipt = json.loads(p.stdout)
    if not receipt["ready"]:
        print(json.dumps({"export_refused_before_large_reads": True, **receipt}))
        return 2
    args.output.mkdir(parents=True)
    (args.output / ".gitignore").write_text("raw/\nraw.tar.gz\n")
    # Re-check native completion before streaming exactly the enumerated files.
    program = ("files=" + repr(receipt["files"]) + "\n" + r'''
import json,pathlib,sys,tarfile
r=json.loads(pathlib.Path('/home/swank/projects/spark-ds41f/results/private/lab/candidate-40-round-attempt-2/receipt.json').read_text())
e=json.loads(pathlib.Path('/home/swank/tl-logs/topk-live-qualification/setup/exit-attempt-2.json').read_text())
assert r.get('completed') and r.get('restored') and r.get('ended') and not r.get('error') and not r.get('restore_error') and e['actual_exit']==0
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as t:
 for name,record in sorted(files.items()):
  t.add(record['source'],arcname=name,recursive=False)
''')
    archive = args.output / "raw.tar.gz"
    with archive.open("xb") as handle:
        p = ssh(program, stdout=handle, stderr=subprocess.PIPE, timeout=180)
    if p.returncode:
        raise SystemExit(p.stderr.decode())
    raw = args.output / "raw"
    raw.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(raw, filter="data")
    actual = {str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()}
    if actual != set(receipt["files"]):
        raise SystemExit("archive and native manifest file sets differ")
    for name, record in receipt["files"].items():
        data = (raw / name).read_bytes()
        if len(data) != record["size"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise SystemExit("native artifact changed or hash differs: " + name)
    receipt.update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   native_file_hashes_verified=True)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified_files": len(actual), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
