
import json,subprocess,sys,os
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/swank/projects/spark-ds41f')
setup=Path('/home/swank/tl-logs/copy-overlap-rework/setup')
log=setup.parent/'candidate-41-window-attempt-2.log'
commit='0da35cc9668ff491065446b2a842fe996a73e80f'
command=[sys.executable,'-u','experiments/2026-11-04-copy-overlap-rework/run_round.py','--apply','--attempt','2','--resume-image-gate-from','results/private/lab/candidate-41-round/receipt.json']
def stamp():return datetime.now(timezone.utc).isoformat()
def save(name,data):
 with (setup/name).open('x') as f:
  json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with log.open('xb') as handle:
 child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,close_fds=True)
 save('launch-attempt-2.json',{'started':stamp(),'controller_pid':child.pid,'supervisor_pid':os.getpid(),'command':command,'cwd':str(root),'log':str(log),'commit':commit})
 code=child.wait()
 save('exit-attempt-2.json',{'ended':stamp(),'actual_exit':code,'controller_pid':child.pid,'supervisor_pid':os.getpid(),'commit':commit})
raise SystemExit(code)
