import pathlib,subprocess
root=pathlib.Path('/home/swank/spark-lab/collective-pdl')
source=root/'sparknet'
for phase, marker in [('staged','                # 3. block 0 waits'),('arrived','                # 4.')]:
    path=root/f'sparknet-{phase}'
    if path.exists(): raise RuntimeError(f'Preserve existing alternative: {path}')
    subprocess.run(['git','worktree','add','-b',f'feat/oneshot-pdl-{phase}',str(path),'fb027b4b80e5262416bb6db2b95b671498f6de74'],cwd=source,check=True)
    for name in ['_oneshot_tilelang.py','_allgather_tilelang.py']:
        file=path/'sparknet/oneshot'/name
        text=file.read_text()
        assert text.count('                T.pdl_trigger()')==1
        text=text.replace('                T.pdl_trigger()\n','')
        assert text.count(marker)==1
        pos=text.index(marker)
        text=text[:pos]+f'                # Experimental {phase} trigger: keep the peer-wait window available to side work.\n                T.pdl_trigger()\n'+text[pos:]
        file.write_text(text)
    file=path/'tests/test_oneshot_cpu.py';text=file.read_text().replace('calls[:4]','calls[:3]').replace('"T.pdl_sync", "T.pdl_trigger"], module_name)','"T.pdl_sync"], module_name)');file.write_text(text)
    file=path/'tests/gpu/test_oneshot_gpu.py';text=file.read_text().replace('assert wait < trigger < epoch','assert wait < epoch and wait < trigger')
    boundary='roce_doorbell(' if phase=='staged' else 'roce_wait_peers('
    text=text.replace('assert wait < epoch and wait < trigger',f'assert wait < epoch and wait < trigger\n            assert trigger > body.index({boundary!r})')
    file.write_text(text)
    file=path/'docs/provenance.md';text=file.read_text();text+=f'\nExperimental launch-placement arm: trigger after {phase} phase, over source fb027b4. Only trigger placement changes; tensor access still waits before the epoch, with unchanged transport, arithmetic and timeouts. This arm is unqualified and is retained for comparison under the collective launch experiment.\n';file.write_text(text)
    subprocess.run(['make','check'],cwd=path,check=True)
    subprocess.run(['git','add','sparknet/oneshot','tests','docs/provenance.md'],cwd=path,check=True)
    subprocess.run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','commit','-m',f'Compare collective launch trigger after the {phase} phase','-m','Base: fb027b4b80e5262416bb6db2b95b671498f6de74. One scheduling variable; native arithmetic, both rails, wire layout and fail-stop paths are unchanged. Tests: make check with the sanitized protocol simulator. GPU tests, fabric probes and performance remain unqualified. Experimental alternative retained for collective launch comparison, not promoted.'],cwd=path,check=True)
    print(phase,subprocess.check_output(['git','rev-parse','HEAD','HEAD^{tree}'],cwd=path,text=True),flush=True)
