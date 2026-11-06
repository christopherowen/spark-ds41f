#!/usr/bin/env python3
"""Scoped DGX1–3 setup and exact C36 recovery for the lossless experiment."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,threading,time
ROOT=pathlib.Path("/home/swank/projects/spark-ds41f")
HOLDER="spark-lab: owner-authorized lossless memory qualification"
RESTORE="experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
CANDIDATE="experiments/2026-11-04-lossless-memory/engram/tp3.json"
SOURCE_PATH="experiments/2026-11-04-lossless-memory/scoped_window.py"
NAMES={"dgx1","dgx2","dgx3"}

def owns(hold,window):
    def same_nodes(value):
        return (isinstance(value,list) and len(value)==3
                and all(isinstance(n,str) for n in value) and set(value)==NAMES)
    if not isinstance(hold,dict):
        return False
    scope=hold.get("resource_scope")
    if not isinstance(scope,dict) or not isinstance(scope.get("restore"),dict):
        return False
    restore=scope["restore"]
    return bool(hold.get("holder")==HOLDER and hold.get("window")==window
                and hold.get("production_config")==RESTORE and same_nodes(hold.get("nodes"))
                and scope.get("kind")=="nodes" and scope.get("holder")==HOLDER
                and scope.get("token")==window and same_nodes(scope.get("nodes"))
                and restore.get("profile")==RESTORE and same_nodes(restore.get("nodes")))


HOLD_PROGRAM = """import json,pathlib
h=pathlib.Path.home()
print(json.dumps([{"name":name,"exists":(h/name).exists(),
                  "value":json.loads((h/name).read_text()) if (h/name).exists() else None}
                 for name in ("spark-hold.json","spark3-hold.json")]))
"""

def node_hold_observations(spark,nodes,window):
    rows,problems=[],[]
    for node in nodes["nodes"]:
        process=spark.run_ssh(nodes,node,"python3","-c",HOLD_PROGRAM)
        row={"node":node["name"],"exit":process.returncode,
             "stdout":process.stdout,"stderr":process.stderr}
        rows.append(row)
        try:
            if process.returncode or len((process.stdout+process.stderr).encode())>65536:
                raise ValueError("unreadable node holds")
            values=json.loads(process.stdout)
            if not isinstance(values,list) or len(values)!=2:
                raise ValueError("incomplete node hold observation")
            if {v["name"] for v in values}!={"spark-hold.json","spark3-hold.json"}:
                raise ValueError("unexpected hold names")
            if any(type(v["exists"]) is not bool or v["exists"] and not owns(v["value"],window)
                   for v in values):
                raise ValueError("foreign or malformed participating-node hold")
        except (ValueError,KeyError,TypeError) as error:
            problems.append(node["name"]+": "+str(error))
    return rows,problems


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("open","watchdog","close","exec"))
    p.add_argument("--source-commit",required=True)
    p.add_argument("--restore-commit",required=True)
    p.add_argument("--window")
    p.add_argument("--out",type=pathlib.Path,required=True)
    p.add_argument("--apply",action="store_true")
    raw=sys.argv[1:]
    divider=raw.index("--") if "--" in raw else len(raw)
    args=p.parse_args(raw[:divider])
    args.command=raw[divider+1:]
    for revision in (args.source_commit,args.restore_commit):
        if len(revision)!=40 or any(c not in "0123456789abcdef" for c in revision):p.error("full published commits required")
    if not args.apply:
        print(json.dumps({"mode":args.mode,"nodes":sorted(NAMES),"restore":RESTORE,
                          "restore_commit":args.restore_commit,"source_commit":args.source_commit,"plan_only":True}));return 0
    if os.uname().nodename!="dgx1":raise SystemExit("wrong host; no operation")
    sys.path.insert(0,str(ROOT))
    import spark_ds41f as spark
    from spark_ds41f import lab
    original=subprocess.check_output(["git","show",args.source_commit+":"+SOURCE_PATH],cwd=ROOT)
    if pathlib.Path(__file__).read_bytes()!=original:raise SystemExit("controller differs from published source")
    lab.PRODUCTION_CONFIG=RESTORE
    cluster,nodes,_=spark.configuration(argparse.Namespace(cluster_config=RESTORE))
    if {n["name"] for n in nodes["nodes"]}!=NAMES:raise SystemExit("wrong physical node pool")
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/"results/private"):p.error("private output required")
    out.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,str(pathlib.Path(__file__).resolve()),"--source-commit",args.source_commit,
          "--restore-commit",args.restore_commit,"--out",str(out.relative_to(ROOT)),"--apply"]
    heartbeat_stop,heartbeat_thread=None,None
    def strict():
        h=lab.read_hold()
        if not owns(h,args.window):raise RuntimeError("foreign, missing or overlapping lease")
        if h["resource_scope"]["restore"]["commit"]!=args.restore_commit:raise RuntimeError("restore identity changed")
        rows,problems=node_hold_observations(spark,nodes,args.window)
        if problems:
            if heartbeat_stop is not None:
                heartbeat_stop.set();heartbeat_thread.join(timeout=5)
            h.update(holder="lossless-memory-ownership-review",
                     rule="foreign or unreadable participating-node hold; preserve scopes and review")
            h["resource_scope"]["holder"]=h["holder"]
            lab.write_json_atomic(lab.HOLD,h)
            lab.write_json_atomic(out/("node-holds-"+str(time.monotonic_ns())+".json"),
                                  {"observations":rows,"problems":problems})
            raise RuntimeError("; ".join(problems))
        return h
    def record_command(argv,name):
        strict()
        path=out/name
        with path.open("x") as log:code=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        lab.write_json_atomic(path.with_suffix(path.suffix+".json"),{"argv":argv,"exit":code,"ended":lab.now()})
        if code:raise RuntimeError(name+" failed")
        return path.read_text(errors="replace")
    def action(profile,operation,*extra):
        argv=[sys.executable,"bin/spark","--cluster-config",profile,"cluster",operation,*extra]
        stamp=str(time.monotonic_ns())
        text=record_command(argv,stamp+"-"+operation+"-plan.log")
        if "--apply would refuse" in text or "shape:   CHANGED" in text:raise RuntimeError("native plan refused")
        return record_command([*argv,"--apply"],stamp+"-"+operation+"-apply.log")
    old_restore=lab.restore_production
    def exact_restore(dry=False):
        if dry:return old_restore(dry=True)
        strict()
        if not lab.production_live():
            # Both profiles are published, same-topology candidates. Native stop
            # refuses unexpected/mixed state rather than launching one rank.
            action(CANDIDATE,"stop","--parallel")
        action(RESTORE,"sync","--revision",args.restore_commit)
        strict()
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.restore_commit:
            raise RuntimeError("head deployment was not restored")
        return old_restore(dry=False)
    if args.mode=="open":
        if lab.read_hold() is not None:raise SystemExit("foreign hold; no acquisition")
        request=lab.REQUEST.read_bytes()
        if json.loads(request).get("who")!="lossless-memory-engram":raise SystemExit("not this task's turn")
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.restore_commit:
            raise SystemExit("capture the actual original deployment before opening")
        refs=subprocess.check_output(["git","branch","--remotes","--contains",args.source_commit,
                                      "--format=%(refname:short)"],cwd=ROOT,text=True).splitlines()
        if not any(ref.startswith("origin/") and ref!="origin/HEAD" for ref in refs):
            raise SystemExit("controller source is not published on a fetched origin branch")
        problems=lab.published_problems()
        if problems:raise SystemExit("publish guard: "+"; ".join(problems))
        rows,problems=node_hold_observations(spark,nodes,None)
        lab.write_json_atomic(out/"pre-open-node-holds.json",{"observations":rows,"problems":problems})
        if problems:raise SystemExit("participating node has an existing or unreadable hold")
        if not lab.production_live() or not lab.idle_for(30):raise SystemExit("native live/idle guard failed")
        if lab.read_hold() is not None or lab.REQUEST.read_bytes()!=request:raise SystemExit("ownership/request changed during capture")
        hold=lab.new_hold(0,"owner-authorized lossless memory qualification")
        args.window=hold["window"]
        watchdog_argv=[base[0],base[1],"watchdog",*base[2:],"--window",args.window]
        with (out/"watchdog.log").open("x") as log:
            watchdog=subprocess.Popen(watchdog_argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL,start_new_session=True)
        hold["nodes"]=sorted(NAMES)
        hold["resource_scope"]={"kind":"nodes","nodes":sorted(NAMES),"holder":HOLDER,"token":args.window,
            "controller":{"commit":args.source_commit,"path":SOURCE_PATH,"pid":os.getpid()},
            "restore":{"profile":RESTORE,"commit":args.restore_commit,
                       "image":"sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11","nodes":sorted(NAMES)},
            "watchdog":{"pid":watchdog.pid,"argv":watchdog_argv,"source_commit":args.source_commit,"nodes":sorted(NAMES)}}
        # Publish a complete scoped hold atomically, without replacing a foreign
        # reservation or exposing a temporary global exclusion.
        temporary=out/"scoped-hold.json"
        lab.write_json_atomic(temporary,hold)
        try:os.link(temporary,lab.HOLD)
        except FileExistsError:
            watchdog.terminate();raise SystemExit("a foreign hold won; no acquisition")
        if lab.REQUEST.read_bytes()!=request:raise RuntimeError("request changed; preserve lease for review")
        os.rename(lab.REQUEST,out/"claimed-request.json")
        lab.write_json_atomic(out/"opened.json",hold)
        print(json.dumps(hold,indent=2));return 0
    if not args.window:p.error("window required")
    if args.mode=="watchdog":
        # The child starts before the complete hold is linked into place.
        until=time.monotonic()+60
        while lab.read_hold() is None and time.monotonic()<until:time.sleep(1)
        lab.hold_is_ours=lambda h: owns(h,args.window)
        lab.restore_production=exact_restore
        try:
            lab.watchdog(900,False)
            if owns(lab.read_hold(),args.window):
                raise RuntimeError("watchdog returned without native restoration and lease release")
        except BaseException as error:
            if owns(lab.read_hold(),args.window):
                h=lab.read_hold();h.update(holder="lossless-memory-restoration-review",
                    rule="watchdog restoration failed; retain exact node scope and review before launch")
                h["resource_scope"]["holder"]=h["holder"]
                lab.write_json_atomic(lab.HOLD,h)
            lab.write_json_atomic(out/"watchdog-error.json",{"error":type(error).__name__+": "+str(error),"ended":lab.now()})
            raise
        return 0
    strict()
    if args.mode=="close":
        lab.hold_is_ours=lambda h: owns(h,args.window)
        lab.restore_production=exact_restore
        try:
            if not lab.window_close("lossless scoped setup finished or failed"):
                raise RuntimeError("native restoration verdict is false")
            return 0
        except BaseException as error:
            if owns(lab.read_hold(),args.window):
                h=lab.read_hold();h.update(holder="lossless-memory-restoration-review",
                    rule="restoration failed; preserve exact node scope for review")
                h["resource_scope"]["holder"]=h["holder"]
                lab.write_json_atomic(lab.HOLD,h)
            lab.write_json_atomic(out/"close-error.json",{"error":type(error).__name__+": "+str(error)})
            raise
    argv=args.command
    if argv and argv[0]=="--":argv=argv[1:]
    if not argv:p.error("exec requires an explicit command")
    h=strict();h["resource_scope"]["controller"].update(pid=os.getpid(),path=SOURCE_PATH,command=argv)
    lab.write_json_atomic(lab.HOLD,h)
    done=threading.Event()
    def heartbeat():
        while not done.wait(30):
            if owns(lab.read_hold(),args.window):lab.beat("lossless Engram setup: "+str(argv[:4]))
    thread=threading.Thread(target=heartbeat,daemon=True)
    heartbeat_stop,heartbeat_thread=done,thread
    thread.start()
    try:
        lab.beat("lossless Engram setup: "+str(argv[:4]))
        name="setup-"+str(time.monotonic_ns())+".log"
        record_command(argv,name);print(json.dumps({"exit":0,"log":str(out/name)}));return 0
    finally:done.set();thread.join(timeout=5)

if __name__=="__main__":raise SystemExit(main())
