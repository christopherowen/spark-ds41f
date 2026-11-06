"""Stream the exact pinned Git archive plus the published Dockerfile to Docker."""
import argparse
import io
from pathlib import Path
import subprocess
import tarfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--head',required=True)
parser.add_argument('--dockerfile',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command
if command and command[0]=='--':command=command[1:]
assert command[:2]==['docker','build'] and command[-1]=='-'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()==args.head
assert not subprocess.check_output(['git','status','--porcelain'],cwd=args.source,text=True).strip()
build=subprocess.Popen(command,stdin=subprocess.PIPE)
archive=subprocess.Popen(['git','archive','--format=tar','--prefix=source/',args.head],cwd=args.source,stdout=subprocess.PIPE)
try:
    with tarfile.open(fileobj=build.stdin,mode='w|') as target:
        with tarfile.open(fileobj=archive.stdout,mode='r|') as source:
            for member in source:
                assert member.name.startswith('source/') and '..' not in Path(member.name).parts
                stream=source.extractfile(member) if member.isfile() else None
                target.addfile(member,stream)
        payload=args.dockerfile.read_bytes()
        member=tarfile.TarInfo('Dockerfile');member.size=len(payload);member.mode=0o644;member.mtime=0
        target.addfile(member,io.BytesIO(payload))
    build.stdin.close();archive.stdout.close()
    assert archive.wait()==0,'Git archive failed'
    raise SystemExit(build.wait())
finally:
    for process in (archive,build):
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
