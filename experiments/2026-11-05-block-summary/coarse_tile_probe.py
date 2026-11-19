"""Out-of-band exact-score tile-size probe; does not change serving defaults."""
import argparse
import json
from pathlib import Path
import runpy
import sys

from vllm.models.deepseek_v4_1.tilelang import block_shortlist

parser = argparse.ArgumentParser()
parser.add_argument('--tile',type=int,choices=[32,64,128],required=True)
parser.add_argument('--action',choices=['check','timing'],required=True)
args,remaining = parser.parse_known_args()
original = block_shortlist.block_scores
block_shortlist.block_scores = lambda width,page,budget:original(width,page,budget,block_c=args.tile)
print('PROBE_CONFIG '+json.dumps({'coarse_tile':args.tile,'runtime_default_unchanged':True}),flush=True)
script = 'check_runtime.py' if args.action == 'check' else 'benchmark_runtime.py'
sys.argv = [script,*remaining]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
