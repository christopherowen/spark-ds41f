# Build one candidate: stack PR patches on 1107 in a scratch vLLM clone, export them, write the candidate folder.
import json,os,posixpath,hashlib,subprocess,re,sys
S,WT,num_c,shape,order=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],[int(x) for x in sys.argv[5].split(",")]
R="/Users/christopherowen/projects/spark-ds41f"; V=f"{S}/vllm"
C="experiments/2026-10-26-spark-ds41f-candidate"; B=f"{C}/1107/vllm"; ND=f"{C}/candidate-{num_c}"; N=f"{ND}/vllm"
def run(*a,cwd=V,env=None): return subprocess.run(list(a),capture_output=True,text=True,cwd=cwd,env=env)
pp=json.load(open(f"{S}/vpatches.json")); prs={p['number']:p for p in json.load(open(f"{S}/prs.json"))}
env=dict(os.environ,GIT_COMMITTER_NAME="b",GIT_COMMITTER_EMAIL="b@b",GIT_COMMITTER_DATE="2026-01-01T00:00:00Z")
run("git","am","--abort"); run("git","checkout","-q","-B",f"c{num_c}","60ca7dac1")
log=[f"candidate-{num_c} stack on spark-ds41f-1107 (patch tip 60ca7dac1, tree {run('git','rev-parse','60ca7dac1^{tree}').stdout.strip()})"]
src=[]
for n in order:
    for fn in pp[str(n)]:
        r=run("git","-c","user.name=b","-c","user.email=b@b","am","-q","--keep-cr",fn,env=env); how="clean"
        if r.returncode:
            run("git","am","--abort"); r=run("git","-c","user.name=b","-c","user.email=b@b","am","-3","-q","--keep-cr",fn,env=env); how="3-way"
            if r.returncode: print("FAIL",n,fn,r.stdout[-500:],r.stderr[-500:]); run("git","am","--abort"); sys.exit(1)
        src.append((n,fn,how)); log.append(f"#{n} {os.path.basename(fn)}: applied {how}")
tree=run("git","rev-parse","HEAD^{tree}").stdout.strip(); log.append(f"stack tree {tree}")
os.chdir(WT); os.makedirs(N,exist_ok=True); os.makedirs(f"{ND}/evidence",exist_ok=True)
lines=[posixpath.relpath(posixpath.normpath(posixpath.join(B,l.strip())),N) for l in open(f"{B}/series") if l.strip() and not l.startswith("#")]
k=len(src); sel=[]
for i,(n,fn,how) in enumerate(src):
    new=f"{762+i:04d}-"+re.sub(r'^\d{4}-','',os.path.basename(fn))
    data=run("git","format-patch","-1","--stdout",f"HEAD~{k-1-i}").stdout if how!="clean" else open(fn).read()
    open(f"{N}/{new}","w").write(data); lines.append(new)
    head=run("git","rev-parse",f"refs/review/pr-{n}",cwd=R).stdout.strip()
    sel.append(dict(patch=new,pr=n,pr_title=prs[n]['title'],pr_head=head,source_patch=os.path.basename(fn),applied=how,sha256=hashlib.sha256(data.encode()).hexdigest()))
open(f"{N}/series","w").write("\n".join([f"# candidate-{num_c}: spark-ds41f-1107's series (0001..0761, 641 active) plus {k} patches from {len(order)} PRs.","# Paths relative to this file. Provenance: ../evidence/selection.json"]+lines)+"\n")
json.dump(dict(candidate=num_c,shape=shape,base="spark-ds41f-1107",base_tree="ed1bdb7a7684da33d403a055521ac0c4dd7ca670",stack_tree=tree,order=order,patches=sel),open(f"{ND}/evidence/selection.json","w"),indent=1)
open(f"{ND}/evidence/stack.log","w").write("\n".join(log)+"\n")
lock=json.load(open(f"{C}/1107/upstreams.lock.json")); v=lock["sources"]["vllm"]; v["patch_series"]=f"{N}/series"
for x in ("patch_head","expected_tree","patchset_sha256"): v.pop(x,None)
lock.update(source_manifest=f"{ND}/source.json",generated_from_baseline="2026-11-07-spark-ds41f-1107",captured_at="2026-11-09",updated_at="2026-11-09")
s=json.load(open(f"{C}/1107/source.json")); sv=s["vllm"]; sv["patch_series"]=f"{N}/series"; sv["patch_count"]=641+k; s["captured_at"]="2026-11-09"
for x in ("patch_head","expected_tree","patchset_sha256"): sv.pop(x,None)
def dump(o,p): open(p,"w").write(json.dumps(o,indent=2)+"\n")
dump(lock,f"{ND}/upstreams.lock.json"); dump(s,f"{ND}/source.json")
labels={"local.spark-ds41f.vllm.tree":tree,"local.spark-ds41f.tilelang.tree":"e2b001828fd4e38e6a2eee61f818d5083cb1335f","local.spark-ds41f.sparknet.tree":"8cc28a5cd3e683c1f114ed843453e3ce34f323e0"}
name={"tp4-ring":"tp4"}.get(shape,shape)
dump({"notes":f"candidate-{num_c} at {shape}: spark-ds41f-1107 plus PRs {' '.join('#'+str(n) for n in order)} (vLLM {641+k} patches, tree {tree[:12]}); unpromoted candidate",
      "extends":{"config":"releases/spark-ds41f-1107/cluster.json","shape":shape},
      "set":{"container":{"image":f"spark-ds41f:candidate-{num_c}","expected_labels":labels},"upstreams_config":f"{ND}/upstreams.lock.json"}},f"{ND}/{name}.json")
dump({"notes":f"candidate-{num_c} at {shape} with development RPC for the engine grid","extends":{"config":f"{ND}/{name}.json"},"set":{"environment":{"VLLM_SERVER_DEV_MODE":"1"}}},f"{ND}/{name}-dev.json")
dump({"notes":f"candidate-{num_c} build-only profile (built on sp8); no serving authorization implied by this file","extends":{"config":f"{ND}/{name}.json"}},f"{ND}/image-sp8.json")
print(f"candidate-{num_c}: {k} patches, tree {tree}"); print("\n".join(log[1:]))
