"""Native spark bench with untimed request-identity and raw-latency receipts.

Payloads are read for hashing, never changed. Cache salts are excluded from the
matching key because they deliberately prevent prefix reuse and do not change
model tokens. A second hash identifies the exact full API payload. No prompt or
activation text is saved by this wrapper.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from spark_ds41f import bench
from spark_ds41f.cli import main


def run():
    manifest=Path(sys.argv[1])
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():raise RuntimeError('refusing existing request manifest')
    original=bench.Bench.run
    def measured(self,label,payloads,*args,**kwargs):
        identities=[]
        for endpoint,payload in payloads:
            def digest(value):
                return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            identities.append({'endpoint':endpoint,'payload_sha256':digest(payload),
                'model_input_sha256':digest({k:v for k,v in payload.items() if k!='cache_salt'})})
        # All hashing happens before the native timer/HTTP workload begins.
        result=original(self,label,payloads,*args,**kwargs)
        requests=[]
        for identity,request in zip(identities,result['requests']):
            requests.append(dict(identity,**{key:request.get(key) for key in [
                'ok','elapsed_s','ttft_s','first_s','last_s','prompt_tokens','completion_tokens',
                'decode_tps','output_sha256','finish_reason','error']}))
        with manifest.open('a') as handle:
            handle.write(json.dumps({'label':label,'wall_s':result['wall_s'],'requests':requests})+'\n')
        return result
    bench.Bench.run=measured
    return main(sys.argv[2:])


if __name__=='__main__':raise SystemExit(run())
