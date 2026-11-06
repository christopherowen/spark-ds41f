"""Stream the exact pinned Git archive plus the published Dockerfile to Docker."""
import argparse
import io
from pathlib import Path
import subprocess
import tarfile
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--head',required=True)
parser.add_argument('--dockerfile',type=Path,required=True)
parser.add_argument('--inspect',action='store_true',help='CPU-only stream verification; no Docker execution')
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command
if command and command[0]=='--':command=command[1:]
if args.inspect:
    assert not command
    command=[sys.executable,str(Path(__file__).with_name('inspect_image_context.py')),str(args.source),args.head,str(args.dockerfile)]
else:
    assert command[:2]==['docker','build'] and command[-1]=='-'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()==args.head
assert not subprocess.check_output(['git','status','--porcelain'],cwd=args.source,text=True).strip()
build=subprocess.Popen(command,stdin=subprocess.PIPE)
archive=subprocess.Popen(['git','-c','tar.umask=0022','archive','--format=tar','--prefix=source/',args.head],cwd=args.source,stdout=subprocess.PIPE)
try:
    with tarfile.open(fileobj=build.stdin,mode='w|') as target:
        # A plain first header lets Docker's 1024-byte archive probe succeed.
        payload=args.dockerfile.read_bytes()
        member=tarfile.TarInfo('Dockerfile');member.size=len(payload);member.mode=0o644;member.mtime=0
        target.addfile(member,io.BytesIO(payload))
        with tarfile.open(fileobj=archive.stdout,mode='r|') as source:
            for member in source:
                assert (member.name=='source' and member.isdir() or member.name.startswith('source/')) and '..' not in Path(member.name).parts
                stream=source.extractfile(member) if member.isfile() else None
                target.addfile(member,stream)
    build.stdin.close();archive.stdout.close()
    assert archive.wait()==0,'Git archive failed'
    raise SystemExit(build.wait())
finally:
    for process in (archive,build):
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
