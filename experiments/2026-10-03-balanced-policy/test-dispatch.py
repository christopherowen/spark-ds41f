"""Run actual B12X dispatch predicates without importing CUDA dependencies."""
from pathlib import Path
import ast,types,argparse,importlib.util,importlib.machinery
ROOT=Path(__file__).resolve().parents[2]
l=importlib.machinery.SourceFileLoader('dispatch_build',str(ROOT/'bin/spark3'));s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
_,_,lock=m.configuration(argparse.Namespace(cluster_config='experiments/2026-10-03-balanced-policy/selected.json'))
i=m.build_inputs(lock);tree=m.build_directory(i)/'src/b12x';assert not m.project_problems(tree,i['b12x'])
s=(tree/'b12x/comm/roce/roce_oneshot.py').read_text();mod=ast.parse(s);cls=next(n for n in mod.body if isinstance(n,ast.ClassDef) and n.name=='RoceOneshotAllReduce')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('should_allreduce','_accepts_allreduce')]
ns={'torch':types.SimpleNamespace(Tensor=object),'SUPPORTED_DTYPES':{'bf16','fp32'},'PACK_BYTES':16}
for n in methods:exec(compile(ast.Module(body=[n],type_ignores=[]),'<actual-dispatch>','exec'),ns)
C=type('Runtime',(),{n.name:ns[n.name] for n in methods});r=C();r._closed=False;r._proxy=object();r.device='gpu0';r.max_size=2097152;r.dispatch_max_bytes=1048576
class Tensor:
 dtype='bf16';is_cuda=True;device='gpu0'
 def __init__(self,n):self.n=n
 def is_contiguous(self):return True
 def numel(self):return self.n//2
 def element_size(self):return 2
for n,expected in [(0,False),(2,False),(16,True),(1048560,True),(1048576,True),(1048592,False),(2097152,False),(2097168,False)]:assert r.should_allreduce(Tensor(n))==expected
assert r._accepts_allreduce(Tensor(2097152),r.max_size)
assert not r._accepts_allreduce(Tensor(2097168),r.max_size)
r.dispatch_max_bytes=r.max_size;assert r.should_allreduce(Tensor(2097152))
r._closed=True
try:r.should_allreduce(Tensor(16))
except RuntimeError:pass
else:raise AssertionError('closed runtime accepted input')
# Explicit priming/launches retain capacity, and peers compare the dispatch limit.
launch=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_prepared_all_reduce')
assert 'self._accepts_allreduce(inp, self.max_size)' in ast.unparse(launch)
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
config=next(n.value for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets))
assert any(isinstance(k,ast.Constant) and k.value=='dispatch_max_bytes' for k in config.keys)
assert '#define ROCE_ABI_VERSION 10' in (tree/'b12x/comm/roce/_roce_proxy.c').read_text()
print('PASS: actual dispatch boundary, independent capacity/priming, unchanged default, closed-runtime rejection, peer agreement field and ABI-10 mixed-version guard')
