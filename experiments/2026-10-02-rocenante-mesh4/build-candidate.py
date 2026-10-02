#!/usr/bin/env python3
"""Build the transport-only image from a verified installed base and source tree."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
loader=importlib.machinery.SourceFileLoader('spark3_mesh_build',str(ROOT/'bin/spark3'))
spec=importlib.util.spec_from_loader(loader.name,loader)
spark3=importlib.util.module_from_spec(spec);loader.exec_module(spark3)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-id',required=True)
a=p.parse_args()
cluster,nodes,lock=spark3.configuration(argparse.Namespace(cluster_config='experiments/2026-10-02-rocenante-mesh4/qualification-mesh.json'))
revision=spark3.require_local_deployment_state(cluster,nodes,lock)
base=spark3.read_json('config/cluster.json')['container']
image=json.loads(subprocess.check_output(['docker','image','inspect',base['image']],text=True))[0]
if image['Id'] != a.base_id:raise SystemExit('base image ID differs from qualification inventory')
for key,value in base['expected_labels'].items():
 if image['Config']['Labels'].get(key)!=value:raise SystemExit('base source label mismatch: '+key)
if subprocess.check_output(['docker','ps','-q'],text=True).strip():raise SystemExit('build requires idle Docker host')
if subprocess.run(['docker','image','inspect',cluster['container']['image']],capture_output=True).returncode==0:raise SystemExit('candidate tag already exists')
inputs=spark3.build_inputs(lock);source=spark3.build_directory(inputs)/'src/b12x'
errors=spark3.project_problems(source,inputs['b12x'])
if errors:raise SystemExit('\n'.join(errors))
manifest=spark3.read_json(lock['source_manifest'])['b12x']
with tempfile.TemporaryDirectory(prefix='mesh4-build-') as temp:
 export=Path(temp)/'b12x';export.mkdir()
 archive=subprocess.Popen(['git','-C',str(source),'archive','HEAD'],stdout=subprocess.PIPE)
 subprocess.run(['tar','-xf','-','-C',str(export)],stdin=archive.stdout,check=True)
 archive.stdout.close()
 if archive.wait():raise SystemExit('source export failed')
 subprocess.run(['docker','buildx','build','--load','--network=none',
  '--build-context','b12x-source='+str(export),
  '--build-arg','BASE_IMAGE='+base['image'],
  '--build-arg','B12X_TREE='+manifest['expected_tree'],
  '--build-arg','B12X_HEAD='+manifest['patch_head'],
  '--build-arg','DEPLOYMENT_COMMIT='+revision,
  '--file',str(Path(__file__).with_name('Dockerfile')),
  '--tag',cluster['container']['image'],str(export)],check=True,timeout=600)
result=json.loads(subprocess.check_output(['docker','image','inspect',cluster['container']['image']],text=True))[0]
for key,value in cluster['container']['expected_labels'].items():
 if result['Config']['Labels'].get(key)!=value:raise SystemExit('candidate label mismatch: '+key)
print(json.dumps({'base_id':image['Id'],'image_id':result['Id'],'revision':revision,'b12x':manifest},indent=2))
