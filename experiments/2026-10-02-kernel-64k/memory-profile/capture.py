"""Read-only host/process memory inventory. Run as root; no CUDA context created."""
import json, os, pathlib, subprocess, sys, time
P = pathlib.Path
out = P(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
paths = ['/proc/meminfo', '/proc/vmstat', '/proc/buddyinfo', '/proc/zoneinfo',
         '/proc/slabinfo', '/proc/vmallocinfo', '/proc/iomem',
         '/sys/kernel/debug/dma_buf/bufinfo', '/proc/driver/nvidia/params']
errors = {}
for name in paths:
    try: (out / name.strip('/').replace('/', '__')).write_text(P(name).read_text())
    except (OSError, UnicodeError) as e: errors[name] = str(e)
processes = []
for proc in P('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        comm = (proc/'comm').read_text().strip()
        rollup = (proc/'smaps_rollup').read_text()
        values = {s.split(':')[0]: int(s.split()[1]) for s in rollup.splitlines()[1:] if ':' in s}
        cg = (proc/'cgroup').read_text().strip()
        processes.append({'pid':int(proc.name), 'comm':comm, 'cgroup':cg, **values})
        if 'docker' in cg:
            (out/('smaps-'+proc.name+'-'+comm.replace('/','_'))).write_text((proc/'smaps').read_text())
    except (OSError, ValueError): pass
processes.sort(key=lambda x: x.get('Pss',0), reverse=True)
(out/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
for group in P('/sys/fs/cgroup/system.slice').glob('docker-*.scope'):
    for name in ['memory.current','memory.stat','memory.events','memory.swap.current']:
        try: (out/(group.name+'-'+name)).write_text((group/name).read_text())
        except OSError as e: errors[str(group/name)] = str(e)
identity = {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'kernel':os.uname().release, 'page_size':os.sysconf('SC_PAGE_SIZE'), 'errors':errors}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity))
