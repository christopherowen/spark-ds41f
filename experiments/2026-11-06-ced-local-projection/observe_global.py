"""Lightweight authenticated hold relay; no lease, serving or GPU action exists."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

READ = "from pathlib import Path; p=Path.home()/'spark-hold.json'; print(p.read_text() if p.exists() else 'null')"
WRITE = """from pathlib import Path
import json,os,sys
p=Path.home()/'spark-lab/ced-sp567/global-observation.json'
p.parent.mkdir(parents=True,exist_ok=True)
v=json.load(sys.stdin)
q=p.with_name('.global-observation-'+str(os.getpid()))
q.write_text(json.dumps(v)+chr(10));os.replace(q,p)
"""

def main():
    import shlex
    deadline=time.monotonic()+7200
    previous=None
    while time.monotonic()<deadline:
        began=time.monotonic()
        value={'source':'dgx1:~/spark-hold.json','success':False,
               'observed_at':dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            read=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','dgx1','python3 -'],
                                input=READ,text=True,capture_output=True,timeout=10,check=True)
            value.update(success=True,hold=json.loads(read.stdout))
        except Exception as error:
            value['error']=repr(error)
        value['read_duration_seconds']=time.monotonic()-began
        state=json.dumps({k:v for k,v in value.items() if k not in ('observed_at','read_duration_seconds')},sort_keys=True)
        try:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','sp5','python3 -c '+shlex.quote(WRITE)],
                           input=json.dumps(value),text=True,capture_output=True,timeout=15,check=True)
            if state!=previous:
                print(json.dumps(value),flush=True);previous=state
        except Exception as error:
            print(json.dumps({'at':value['observed_at'],'relay_error':repr(error)}),flush=True)
        time.sleep(3)
    print('relay time limit reached',flush=True)

if __name__=='__main__':main()
