from pathlib import Path
import subprocess,json,hashlib,datetime,ast,shutil
h=Path.home();dev=h/'spark-lab/checkouts/lsa-development';rel='experiments/2026-11-06-wide-coarse';exp=dev/rel;out=h/'spark-lab/results/wide-coarse/prepare-01';base='f48423d50b97878f5c74bc2637ec60dbd7da8d5f';commands=[]
def run(argv,cwd=dev):
    commands.append({'argv':argv,'cwd':str(cwd)})
    return subprocess.run(argv,cwd=cwd,check=True,timeout=120)
def put(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write(chr(10))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip()==base
assert not subprocess.check_output(['git','status','--porcelain'],cwd=dev,text=True).strip()
exp.mkdir();(exp/'overlay').mkdir();(exp/'vllm').mkdir();(exp/'runs/local').mkdir(parents=True)
wide=(dev/'experiments/2026-11-06-wide-merge/overlay/wide_topk_merge.py').read_text()
scan=wide[wide.index('    @T.macro\n    def block_scan'):wide.index('    @T.macro\n    def body')]
start=wide.index('                key = T.alloc_var(T.int32)');end=wide.index('\n    if final:',start)
emit=wide[start:end].replace('WorkEntries[row, base+tx*per+v]','base+tx*per+v')
module='''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact top8192 from raw16384-entry tiles; eliminate two select-all stages."""
import tilelang
import tilelang.language as T
import torch
from . import prepare,pdl
from .indexer import IndexerTopK,_make,_row_extent,_key16,_value16,_select_macros,_final_writer

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC:True})
def raw_wide_pair(width:int):
    rows=T.dynamic('rows')
    topk,threads=8192,512
    span=2*topk;per=span//threads;steps=threads.bit_length()-1
    groups=tilelang.cdiv(width,span);final=groups==1
    threshold,_=_select_macros(threads,topk)
    _,write_empty=_final_writer(threads,topk,1)
'''+scan+'''
    @T.macro
    def body(Scores,Lengths,Active,Indices,Values):
        with T.Kernel(groups,rows,threads=threads) as (g,row):
            pdl.chain()
            tx=T.get_thread_binding()
            keys=T.alloc_shared([span],T.int32)
            hist=T.alloc_shared([257],T.int32)
            scan=T.alloc_shared([threads],T.int32)
            params=T.alloc_shared([4],T.int32)
            extent=_row_extent(Lengths,Active,row,width)
            base=g*span
            if base<extent:
                for i in T.serial(per):
                    j=i*threads+tx
                    keys[j]=T.if_then_else(base+j<extent,
                        _key16(Scores[row,T.min(base+j,width-1)]),0)
                T.sync_threads()
                threshold(keys,span,tx,hist,scan,params)
'''+emit+'''
            elif final:
                write_empty(Indices,Values,None,row)

    @T.prim_func
    def run(Scores:T.Tensor([rows,width],T.bfloat16),Lengths:T.Tensor([rows],T.int32),
            Active:T.Tensor([1],T.int32),Indices:T.Tensor([rows,groups*topk],T.int32),
            Values:T.Tensor([rows,groups*topk],T.bfloat16)):
        body(Scores,Lengths,Active,Indices,Values)
    return run

'''
cls=(dev/'experiments/2026-11-05-direct-coarse-merge/overlay/coarse_merge.py').read_text();cls=cls[cls.index('class DirectCoarseTopK:'):]
cls=cls.replace('DirectCoarseTopK','RawWideCoarseTopK').replace('Raw pairs replace the all-entry segment pass and first merge exactly.','Raw tiles replace the all-entry segment pass and first two merges exactly.').replace('4096','8192').replace('tilelang.cdiv(width,8192)','tilelang.cdiv(width,16384)').replace('segments>=2 and self.groups==levels[0][2]','segments>=3 and self.groups==levels[1][2]').replace('raw_pair(width)','raw_wide_pair(width)').replace('prepare.call(raw_pair,width)','prepare.call(raw_wide_pair,width)').replace('levels[1:]','levels[2:]')
module+=cls;ast.parse(module)
parent=json.loads((dev/'experiments/2026-11-06-wide-merge/runs/local/source-check.json').read_text());lock=json.loads((dev/'experiments/2026-10-26-spark-ds41f-candidate/candidate-36/upstreams.lock.json').read_text())['sources']['vllm'];cached=h/'spark-lab/checkouts/vllm-drafter-orchestration-20261105';fresh=dev/'.work/wide-coarse-source-01'
run(['git','clone','--shared','--no-checkout',str(cached),str(fresh)]);run(['git','remote','rename','origin','upstream'],fresh);run(['git','remote','set-url','upstream',lock['upstream']],fresh);run(['git','remote','add','origin',lock['contribution_remote']],fresh);run(['git','checkout','-b','wide-coarse-source',parent['base']],fresh)
for p,digest in parent['ordered_patches'].items():
    assert hashlib.sha256((dev/p).read_bytes()).hexdigest()==digest
    run(['git','apply',str(dev/p)],fresh)
run(['git','add','vllm'],fresh)
p=fresh/'vllm/models/deepseek_v4_1/tilelang/compact_shortlist.py';s=p.read_text();needle='from .coarse_merge import DirectCoarseTopK';assert s.count(needle)==1;s=s.replace(needle,needle+'\nfrom .raw_wide_coarse import RawWideCoarseTopK')
needle='self.coarse = (DirectCoarseTopK(width//8)';s=s.replace(needle,'self.coarse = (RawWideCoarseTopK(width//8) if self.source and self.block_budget==8192 and width//8>8192\n                       else DirectCoarseTopK(width//8)');assert s.count('self.coarse = (RawWideCoarseTopK')==1
needle='coarse = (DirectCoarseTopK.calls(width//8)';s=s.replace(needle,'coarse = (RawWideCoarseTopK.calls(width//8) if candidate_blocks and b*8==8192 and width//8>8192\n                  else DirectCoarseTopK.calls(width//8)');assert s.count('coarse = (RawWideCoarseTopK.calls')==1
ast.parse(s);p.write_text(s);q=p.with_name('raw_wide_coarse.py');q.write_text(module);run(['git','add','-N',str(q.relative_to(fresh))],fresh)
patch=subprocess.check_output(['git','diff','--binary'],cwd=fresh,text=True)
header='# Upstream base: C36 vLLM 6d4e49d1cdbb6513819e47145e0e52fdab327b6b.\n# Requires the ordered eleven shortlisting, phase and wide-merge patches.\n# Rationale: select top8192 directly from raw16384-entry spans, eliminating\n# the all-entry stage and first all-entry merge before the equivalent second merge.\n# Quality: identical radix thresholds, BF16 values and ascending-entry tie policy;\n# no retention, cache, score arithmetic, phase or answer-policy change.\n# Tests required: independent stable sort, prior-route equality, initialization/race,\n# declared calls,12 frozen native-score batches and72 phase graph routes.\n# Full model answer, repeated request and admission qualification remain required.\n# Upstream status: unpromoted experiment; not submitted upstream.\n'
(exp/'vllm/0001-wide-coarse.patch').write_text(header+patch);(exp/'vllm/series').write_text('0001-wide-coarse.patch\n')
shutil.copyfile(p,exp/'overlay/compact_shortlist.py');shutil.copyfile(q,exp/'overlay/raw_wide_coarse.py')
verify=dev/'.work/wide-coarse-verification-01';run(['git','clone','--shared','--no-checkout',str(cached),str(verify)]);run(['git','remote','rename','origin','upstream'],verify);run(['git','remote','set-url','upstream',lock['upstream']],verify);run(['git','remote','add','origin',lock['contribution_remote']],verify);run(['git','checkout','-b','wide-coarse-verification',parent['base']],verify)
for p in parent['ordered_patches']:run(['git','apply',str(dev/p)],verify)
run(['git','apply','--check',str(exp/'vllm/0001-wide-coarse.patch')],verify);run(['git','apply',str(exp/'vllm/0001-wide-coarse.patch')],verify)
files={**parent['exact_runtime_files'],rel+'/overlay/compact_shortlist.py':'vllm/models/deepseek_v4_1/tilelang/compact_shortlist.py',rel+'/overlay/raw_wide_coarse.py':'vllm/models/deepseek_v4_1/tilelang/raw_wide_coarse.py'}
for p,t in files.items():assert (dev/p).read_bytes()==(verify/t).read_bytes()
for tree in [fresh,verify]:run(['git','diff','--check'],tree)
mounts=json.loads((dev/'experiments/2026-11-06-wide-merge/tp4.json').read_text())['set']['container']['mounts']
for row in mounts:
    if row[1].endswith('/tilelang/compact_shortlist.py'):row[0]='{checkout}/'+rel+'/overlay/compact_shortlist.py'
mounts.append(['{checkout}/'+rel+'/overlay/raw_wide_coarse.py','/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/raw_wide_coarse.py','ro'])
put(exp/'tp4.json',{'extends':{'config':'experiments/2026-11-06-wide-merge/tp4.json'},'set':{'container':{'mounts':mounts}},'notes':'Exact raw16384-to8192 coarse selection removes two select-all stages and quarters coarse ping-pong scratch at full source width. Same source1024/dense64/native-generation policy and all arithmetic. Unpromoted; parent answer quality and own correctness/performance gates remain required.'})
put(exp/'base.json',{'commit':base,'variable':'exact raw wide coarse selection skips redundant all-entry stages','parent_model_quality_pending':True,'new_approximation':False})
patches={**parent['ordered_patches'],rel+'/vllm/0001-wide-coarse.patch':hashlib.sha256((exp/'vllm/0001-wide-coarse.patch').read_bytes()).hexdigest()}
put(exp/'runs/local/source-check.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':parent['base'],'branch':'wide-coarse-verification','ordered_patches':patches,'exact_runtime_files':files,'diff_check':True,'new_approximation':False,'commands':commands,'cpu_quota_percent':100,'memory_limit_gib':4,'complete_image_build_inputs_present':False})
shutil.copyfile(out/'prepare.py',exp/'runs/local/prepare.py')
(exp/'README.md').write_text('# Exact raw wide coarse selection'+chr(10)+chr(10)+'Base `'+base+'`. One intended variable relative to the source1024/dense64/native-generation recipe: replace the all-entry4096-key segment pass and following select-all8192-key merge with a raw16384-key to8192-key selection. Remaining exact wide merges are unchanged. The first useful selection sees the same entries and retains the same BF16 thresholds, score bits and ascending-entry ties. The full-width coarse route removes two launches and reduces its ping-pong scratch by four; these are structural counts, not measured request-speed gains.'+chr(10)+chr(10)+'The parent answer comparison is still running. This optimization cannot repair a parent selection-quality loss. Retention, cache geometry, score arithmetic, phase policy, weights and final-answer protocol remain unchanged. Independent prior-route/stable-sort, strict initialization/race, actual native-score, declared-call and graph checks must pass from a clean published commit. Two fresh component timing processes and independent model request/answer gates remain required before any speed or quality claim.'+chr(10)+chr(10)+'Fresh twelve-patch application equals all affected overlays. Commands and source hashes remain in `runs/local`; preparation uses one CPU/4 GiB and one worker remotely. Complete image-build inputs are unavailable. All reference calculations remain outside serving, raw operands remain private on nodes, and no production promotion is claimed.'+chr(10))
run(['bin/spark','--cluster-config',rel+'/tp4.json','doctor']);run(['git','diff','--check']);run(['git','add',rel]);run(['git','diff','--cached','--check']);run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','commit','-m','Remove redundant selection stages from wide coarse routing'])
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip();bundle=out/'prepared.bundle';run(['git','bundle','create',str(bundle),base+'..HEAD']);put(out/'prepared.json',{'commit':head,'bundle_bytes':bundle.stat().st_size,'source_verified':True,'published':False});print((out/'prepared.json').read_text())
