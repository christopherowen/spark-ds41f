import argparse,json,pathlib,subprocess,sys
root=pathlib.Path('/home/swank/spark-lab/collective-pdl/deployment')
sys.path.insert(0,str(root))
import spark_ds41f as spark
from spark_ds41f import build
profile='experiments/2026-11-05-collective-pdl/tp3-triangle.json'
up=spark.configuration(argparse.Namespace(cluster_config=profile))[2]
inputs=build.build_inputs(up); dest=build.build_directory(inputs)
old=root/'.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-f970f8b7'
assert dest != old
records=[]
for name in ['vllm','b12x','nccl','tilelang','tile_kernels','cutlass']:
    source=old/'src'/name; target=dest/'src'/name
    assert not target.exists(),target
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cp','--reflink=auto','--archive',str(source),str(target)],check=True)
source=old/'context/cutlass-dsl-wheels';target=dest/'context/cutlass-dsl-wheels'
assert not target.exists(),target
target.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['cp','--reflink=auto','--archive',str(source),str(target)],check=True)
print(dest,flush=True)
for command in ['prepare','check']:
    subprocess.run([str(root/'bin/spark'),'--cluster-config',profile,'build',command],cwd=root,check=True)
