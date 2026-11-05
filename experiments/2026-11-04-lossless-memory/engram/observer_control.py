#!/usr/bin/env python3
"""Read-only native permission control before an Engram image is built."""
import argparse,hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments/2026-11-04-lossless-memory"))
import run_component as component
import spark_ds41f as spark
from spark_ds41f import lab

NS_PROGRAM=r'''
import json,pathlib,sys
result={}
for pid in sys.argv[1:]:
 lines=(pathlib.Path("/proc")/pid/"status").read_text().splitlines()
 line=next(x for x in lines if x.startswith("NSpid:"))
 result[pid]=[int(x) for x in line.split()[1:]]
print(json.dumps(result))
'''
INSPECT_FORMAT = '{"Id":{{json .Id}},"Image":{{json .Image}},"State":{"Running":{{json .State.Running}},"StartedAt":{{json .State.StartedAt}}},"User":{{json .Config.User}},"Privileged":{{json .HostConfig.Privileged}},"CapAdd":{{json .HostConfig.CapAdd}}}'
FD_PROGRAM=r'''
import json,os,pathlib,sys
result={}
for pid in sys.argv[1:]:
 root=pathlib.Path("/proc")/pid
 stat=(root/"stat").read_text()
 row={"start_ticks":int(stat[stat.rfind(")")+2:].split()[19]),"fds":[],"closed_during_observation":[]}
 for p in sorted((root/"fd").iterdir(),key=lambda p:int(p.name)):
  try:
   target=os.readlink(p);identity=p.stat()
  except FileNotFoundError:
   row["closed_during_observation"].append(p.name);continue
  row["fds"].append({"fd":int(p.name),"target":target,"device":identity.st_dev,"inode":identity.st_ino})
 assert row["fds"],"serving process has no readable FDs"
 after=(root/"stat").read_text()
 assert row["start_ticks"]==int(after[after.rfind(")")+2:].split()[19]),"PID identity changed"
 result[pid]=row
print(json.dumps({"processes":result,"all_inspected_fds_readable":True}))
'''

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--window",required=True)
 p.add_argument("--out",type=pathlib.Path,required=True)
 p.add_argument("--apply",action="store_true")
 args=p.parse_args();out=(ROOT/args.out).resolve()
 if not out.is_relative_to(ROOT/"results/private"):p.error("control receipts must remain private")
 if not args.apply:
  print(json.dumps({"scope":"CPU-only metadata reads on each TP3 rank; no model requests or serving lifecycle operations","config":component.RESTORE,"window":args.window,"out":str(out),"io_memory_mib":128,"fd_memory_mib":128,"cpus_per_observer":1}));return 0
 def guard():
  h=lab.read_hold()
  if not (h and h.get("holder")==component.HOLDER and h.get("window")==args.window and h.get("production_config")==component.RESTORE):raise RuntimeError("foreign window; no observation")
 if os.uname().nodename!="dgx1" or out.exists():raise SystemExit("wrong host or prior output; no operation")
 guard()
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():raise SystemExit("dirty source; no operation")
 cluster,nodes,_=spark.configuration(argparse.Namespace(cluster_config=component.RESTORE))
 if len(nodes["nodes"])!=3:raise SystemExit("only configured TP3 ranks are authorized")
 out.mkdir(parents=True);receipt={"scope":"native CPU-only /proc I/O and FD permission controls; no serving qualification or baseline timing","window":args.window,"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"started":lab.now(),"nodes":[],"passed":False}
 def save():lab.write_json_atomic(out/"receipt.json",receipt)
 def run(node,*argv):
  guard();process=spark.run_ssh(nodes,node,*argv)
  row={"argv":list(argv),"exit":process.returncode,"stdout":process.stdout,"stderr":process.stderr}
  receipt["nodes"][-1]["steps"].append(row);save()
  if process.returncode or len((process.stdout+process.stderr).encode())>256*1024:raise RuntimeError("native observation failed or exceeded output bound")
  return process.stdout
 try:
  for node in nodes["nodes"]:
   lab.beat("lossless Engram native observer controls: "+node["name"])
   receipt["nodes"].append({"node":node["name"],"steps":[],"passed":False});save()
   inspected=json.loads(run(node,"docker","inspect",cluster["container"]["name"],"--format",INSPECT_FORMAT))
   image=inspected["Image"];assert inspected["State"]["Running"] and inspected["User"] in ("","root","0")
   assert not inspected["Privileged"] and set(inspected.get("CapAdd") or []) <= {"IPC_LOCK","CAP_IPC_LOCK"}
   pids=component.io_process_ids(run(node,"docker","top",cluster["container"]["name"],"-eo","pid"))
   io=json.loads(run(node,*component.io_command(image,cluster["container"]["name"],pids)))
   component.io_gate(io,cluster["container"]["name"],expected_pids=pids)
   ns=json.loads(run(node,"python3","-c",NS_PROGRAM,*pids))
   assert set(ns)==set(pids) and all(ns[pid] and ns[pid][0]==int(pid) and ns[pid][-1]>0 for pid in pids)
   namespace_pids=[str(ns[pid][-1]) for pid in pids];assert len(namespace_pids)==len(set(namespace_pids))
   fdargs=component.io_command(image,cluster["container"]["name"],pids)
   fdargs[fdargs.index("--pid=host")]="--pid=container:"+cluster["container"]["name"]
   i=fdargs.index("-c");fdargs=fdargs[:i]+["-c",FD_PROGRAM,*namespace_pids]
   fd=json.loads(run(node,*fdargs))
   assert fd["all_inspected_fds_readable"] is True and set(fd["processes"])==set(namespace_pids)
   for pid in pids:assert io["processes"][pid]["start_ticks"]==fd["processes"][str(ns[pid][-1])]["start_ticks"]
   after=json.loads(run(node,"docker","inspect",cluster["container"]["name"],"--format",INSPECT_FORMAT))
   assert all(after[key]==inspected[key] for key in ["Id","Image"]) and after["State"]["StartedAt"]==inspected["State"]["StartedAt"] and after["State"]["Running"]
   receipt["nodes"][-1].update(passed=True,image_id=image,serving_id=inspected["Id"],serving_started=inspected["State"]["StartedAt"],host_pids=pids,namespace_pids=namespace_pids,io_program_sha256=hashlib.sha256(component.IO_PROGRAM.encode()).hexdigest(),fd_program_sha256=hashlib.sha256(FD_PROGRAM.encode()).hexdigest(),io=io,fd=fd);save()
  receipt["passed"]=True
 except Exception as error:
  receipt["error"]=type(error).__name__+": "+str(error)
 finally:
  receipt["ended"]=lab.now();save()
 print(json.dumps({"path":str(out/"receipt.json"),"passed":receipt["passed"]}))
 return 0 if receipt["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
