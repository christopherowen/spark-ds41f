"""Independently verify streamed context bytes against Git without running Docker."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

source,head,dockerfile=sys.argv[1:]
expected={}
for line in subprocess.check_output(['git','ls-tree','-r',head],cwd=source,text=True).splitlines():
    meta,path=line.split('\t',1);mode,kind,blob=meta.split();assert kind=='blob'
    expected['source/'+path]=(mode,blob)
seen=set();recipe=False
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
    for member in archive:
        if member.isdir():continue
        if member.name=='Dockerfile':
            assert archive.extractfile(member).read()==Path(dockerfile).read_bytes();recipe=True;continue
        assert member.name in expected and member.name not in seen,member.name
        mode,blob=expected[member.name]
        if mode=='120000':
            assert member.issym();data=member.linkname.encode()
            digest=hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest()
        else:
            assert member.isfile() and (member.mode & 0o777)==int(mode[-3:],8),(member.name,member.mode,mode)
            digest=hashlib.sha1(b'blob '+str(member.size).encode()+bytes([0]))
            with archive.extractfile(member) as stream:
                for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
            digest=digest.hexdigest()
        assert digest==blob,(member.name,digest,blob)
        seen.add(member.name)
assert seen==set(expected) and recipe,(len(seen),len(expected),recipe)
print(json.dumps({'verified_files':len(seen),'source_head':head,'dockerfile_exact':recipe,'scope':'CPU-only exact streaming context check'}))
