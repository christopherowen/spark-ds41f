#!/usr/bin/env python3
"""Wait for other sp8 windows without changing their containers or hold files."""
import argparse
import json
from pathlib import Path
import subprocess
import time
parser=argparse.ArgumentParser()
parser.add_argument('--timeout',type=int,default=7200)
parser.add_argument('runs',nargs='+',help='matrix-name:run-name')
args=parser.parse_args()
here=Path(__file__).resolve().parent
end=time.monotonic()+args.timeout
previous=None
while True:
    reason=None
    for name in ['spark-hold.json','spark3-hold.json']:
        path=Path.home()/name
        if path.exists():reason=json.loads(path.read_text()).get('holder',name)
    if not reason:
        ids=subprocess.check_output(['docker','ps','-q'],text=True).split()
        for identifier in ids:
            data=json.loads(subprocess.check_output(['docker','inspect',identifier],text=True))[0]
            host=data['HostConfig']
            if host['Runtime']!='runc' or host.get('DeviceRequests') or host.get('Devices') or host.get('Privileged'):
                reason=data['Name'];break
    if not reason:break
    if reason!=previous:print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'waiting for',reason,flush=True)
    previous=reason
    if time.monotonic()>end:raise SystemExit('timed out waiting for an idle sp8')
    time.sleep(10)
for item in args.runs:
    matrix,run=item.split(':',1)
    subprocess.run(['python3',str(here/'run_matrix.py'),str(here/f'matrix-{matrix}.json'),'--run',run],check=True)
