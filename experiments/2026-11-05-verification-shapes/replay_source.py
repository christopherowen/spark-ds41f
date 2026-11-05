"""Fresh canonical replay of the experiment's pinned vLLM patch series."""
import argparse,json,pathlib,subprocess
p=argparse.ArgumentParser(); p.add_argument('--cache',type=pathlib.Path,required=True); p.add_argument('--destination',type=pathlib.Path,required=True); p.add_argument('--variant',choices=('row-ladder','shape-union')); p.add_argument('--out',type=pathlib.Path,required=True); a=p.parse_args()
exp=pathlib.Path(__file__).resolve().parent; dep=exp.parents[1]; model=exp/'variants'/a.variant if a.variant else exp; out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False); lock=json.loads((model/'upstreams.lock.json').read_text())['sources']['vllm']; src=a.destination.resolve(); commands=[]
log=(out/'source-replay.txt').open('w')
def run(args):
 args=list(map(str,args)); commands.append(args); subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
def rev(ref): return subprocess.check_output(['git','-C',str(src),'rev-parse',ref],text=True).strip()
try:
 run(['git','clone','--shared','--no-checkout',a.cache,src]); run(['git','-C',src,'checkout','-b','verification-shapes-replay',lock['revision']])
 series=dep/lock['patch_series']; paths=[(series.parent/x.strip()).resolve() for x in series.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
 for i,path in enumerate(paths):
  if i==574:
   assert rev('HEAD')=='5f926df6d3126266a9c88c82c06544060008c33c'
   assert rev('HEAD^{tree}')=='ca1496c6481dc5bc4b4234b743e4ca023a6c7597'
  run(['git','-C',src,'-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','am','--quiet','--committer-date-is-author-date',path])
 assert rev('HEAD')==lock['patch_head']; assert rev('HEAD^{tree}')==lock['expected_tree']
 run(['git','-C',src,'diff','--check',lock['revision'],'HEAD'])
 result={'patch_count':len(paths),'head':rev('HEAD'),'tree':rev('HEAD^{tree}'),'base_reproduced':True,'passed':True}
 (out/'source-replay.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
finally:
 (out/'source-replay-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
