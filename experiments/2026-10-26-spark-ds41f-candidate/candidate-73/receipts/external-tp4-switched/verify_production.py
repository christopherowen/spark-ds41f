import json,pathlib,socket,subprocess,urllib.request
B=pathlib.Path(__file__).resolve().parent
base='http://REDACTED_IP:8888'
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=20))
names=[x['id'] for x in m['data']]
assert 'deepseek-v4.1-flash' in names and 'DeepSeek-V4.1-Flash-1017' in names
assert 'deepseek-v4.1-flash-weightless' not in names
completions={}
for name in names:
    p={'model':name,'messages':[{'role':'user','content':'Reply with exactly OK.'}],'max_tokens':16,'temperature':0,'chat_template_kwargs':{'thinking':False}}
    r=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
    c=json.load(urllib.request.urlopen(r,timeout=90))
    assert c['choices'][0]['message']['content'].strip()=='OK',c
    completions[name]=c
s=socket.socket(); s.settimeout(3)
closed=s.connect_ex(('REDACTED_IP',8889)); s.close()
assert closed!=0
health=json.load(urllib.request.urlopen('http://REDACTED_IP:8642/health',timeout=10))
assert health['status']=='ok'
report={'models':m,'completions':completions,'candidate_port_connect_ex':closed,'gateway_health':health,'nodes':{}}
for n in ['rank0','rank1','rank2','rank3']:
    cmd='cd /home/operator/projects/spark-ds41f && git rev-parse HEAD && test -z "$(git status --porcelain)" && systemctl is-active spark-ds41f-memguard.service && docker inspect spark-ds41f'
    r=subprocess.run(['ssh','-o','BatchMode=yes',n,cmd],capture_output=True,text=True,check=True)
    lines=r.stdout.splitlines()
    assert lines[0]=='cc3fe02cda60bf7e2ca96b170bcc4226fd8a8f4e' and lines[1]=='active'
    obj=json.loads('\n'.join(lines[2:]))[0]
    assert obj['State']['Running'] and obj['RestartCount']==0 and obj['Config']['Image']=='spark-ds41f:1120'
    assert not any('WEIGHTLESS_STEER' in v or 'VLLM_SERVER_DEV_MODE=1' in v for v in obj['Config']['Env'])
    saved=json.loads((B/f'{n}-rollback-image.json').read_text())[0]
    assert obj['Image']==saved['Id']
    facts=subprocess.run(['ssh','-o','BatchMode=yes',n,'free -b; docker logs spark-ds41f'],capture_output=True,text=True,check=True)
    (B/f'{n}-rollback-state.txt').write_text(facts.stdout+facts.stderr)
    memoryline=[x for x in facts.stdout.splitlines() if x.startswith('Mem:')][0]
    available=int(memoryline.split()[-1])
    assert available>5*1024**3
    report['nodes'][n]={'commit':lines[0],'guard':lines[1],'image':obj['Config']['Image'],'image_id':obj['Image'],'restart_count':obj['RestartCount'],'container_id':obj['Id'],'mem_available_bytes':available}
(B/'production-restored.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'production':'1120','aliases_completed':list(completions),'all_nodes_verified':list(report['nodes']),'candidate_port_closed':True,'gateway':health},indent=2))
