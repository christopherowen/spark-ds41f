"""CPU-only checks using pinned dependencies and the full Python source delta."""
import argparse,json,pathlib,subprocess
p=argparse.ArgumentParser(); p.add_argument('--source',type=pathlib.Path,required=True); p.add_argument('--out',type=pathlib.Path,required=True); a=p.parse_args()
src=a.source.resolve(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
exp=pathlib.Path(__file__).resolve().parent; dest='/opt/spark-ds41f/candidate/vllm'
base='1f33102547e4da5c64712390604695d1abd577c4'; image='sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b'
paths=subprocess.check_output(['git','-C',str(src),'diff','--name-only',base],text=True).splitlines()
cmd=['docker','run','--rm','--name','verification-shapes-cpu','--cpus=1','--memory=4g','--network=none','-e','NVIDIA_VISIBLE_DEVICES=void','-e','HF_HUB_OFFLINE=1','-e','VLLM_NO_USAGE_STATS=1','-e','DO_NOT_TRACK=1','-e','OMP_NUM_THREADS=1','-w',dest,'--entrypoint','/bin/sh']
cmd+=['-v',f'{exp}/cpu_bootstrap.py:/tmp/cpu_bootstrap.py:ro']
for name in paths:
 path=src/name
 if not path.is_file(): raise SystemExit(f'Cannot overlay removed path: {name}')
 cmd+=['-v',f'{path}:{dest}/{name}:ro']
cmd += [image,'-c','uv venv --system-site-packages /tmp/.venv && /tmp/.venv/bin/python /tmp/cpu_bootstrap.py']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'output.txt').open('w') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
print((out/'output.txt').read_text()); raise SystemExit(result.returncode)
