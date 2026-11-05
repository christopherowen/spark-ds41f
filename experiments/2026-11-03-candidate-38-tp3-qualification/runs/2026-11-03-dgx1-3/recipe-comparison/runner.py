"""Compare already measured recipes; retain each profile's own repeat boots."""
import hashlib,json,pathlib,sys
repo=pathlib.Path.cwd();sys.path.insert(0,str(repo))
from servebench import suite
from servebench.__main__ import compare_results,read_json
from scripts.servebench_suite_table import AA_FLOOR_PCT
private=pathlib.Path(__file__).resolve().parent.parent
root=private/'snapshots';judge=lambda a,b:compare_results(a,b,3.0)
result={'recorded_runs_only':True,'no_inference_requests':True,'variable':'complete candidate-38 CopyDrafter recipe versus the standard recipe, including its recorded batch budget/capture sizing','method':'Compare each standard boot separately with both copy boots. Only the two copy boots enter their repeat allowance; distinct profiles are never pooled as repeats. Preserve both reference choices.','comparisons':{}}
for suffix in ['', '-single-stream', '-sampled']:
 base_dir=root/'release-default/raw/results/private/suites'/('candidate-38-tp3-qualification-default'+suffix)
 candidate_dir=root/'release-copy/raw/results/private/suites'/('candidate-38-tp3-qualification-copy'+suffix)
 a=candidate_dir/'candidate-38-copy';b=candidate_dir/'candidate-38-copy-repeat';aa=suite.compare(read_json,str(a),str(b),judge)
 for ref in ['candidate-38','candidate-38-repeat']:
  base=base_dir/ref;x=suite.compare(read_json,str(base),str(a),judge);y=suite.compare(read_json,str(base),str(b),judge)
  result['comparisons'][ref+suffix]={'standard_to_copy_boot1':x,'standard_to_copy_boot2':y,'headlines':suite.with_floor(x,aa,floors=AA_FLOOR_PCT,reference_to_repeat=y)}
p=private/'recipe-comparison.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
for name,c in result['comparisons'].items():
 print(name)
 for row in c['headlines']:
  if row['metric']!='output tok/s':print(row['step'],row['metric'],round(row['change_pct'],3),'margin',round(row['margin_pct'],3),row['verdict'])
print('saved',p,'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
