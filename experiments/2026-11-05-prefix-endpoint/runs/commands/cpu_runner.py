import json, os, resource, sys, time
import pytest
started=time.monotonic()
code=pytest.main(sys.argv[1:])
import torch
print('CPU_RECEIPT '+json.dumps({'exit_code':int(code),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_s':time.monotonic()-started,'cuda_initialized':torch.cuda.is_initialized(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_limit':2,'memory_limit_gib':4}),flush=True)
raise SystemExit(code)
