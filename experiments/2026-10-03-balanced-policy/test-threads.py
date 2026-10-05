"""Test the actual tiny-Ring thread helper extracted from prepared NCCL."""
from pathlib import Path
import subprocess,tempfile,importlib.machinery,importlib.util,argparse
ROOT=Path(__file__).resolve().parents[2]
l=importlib.machinery.SourceFileLoader('thread_build',str(ROOT/'bin/spark'));s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
_,_,lock=m.configuration(argparse.Namespace(cluster_config='experiments/2026-10-03-balanced-policy/adaptive.json'))
i=m.build_inputs(lock);tree=m.build_directory(i)/'src/nccl';assert not m.project_problems(tree,i['nccl'])
s=(tree/'src/enqueue.cc').read_text();a=s.index('static void balanceSwitchlessThreads(');b=s.index('\n}\n',a)+3
code='#include <cassert>\n#include <cstdint>\n#include <cstddef>\n#include <initializer_list>\nusing std::size_t;\n'+s[a:b]+r"""
int main(){
 for(int mode:{0,1,2}) for(int nc:{1,2,4,8,16}) for(int th:{1,8,32,64})
 for(int start:{64,128,256,512}) for(size_t bytes=1;bytes<1048576;bytes+=17){
  int nt=start;balanceSwitchlessThreads(mode,bytes,nc,th,&nt);
  assert(nt<=start && nt>=64 && (nt==start || nt>=128));
  if(mode!=2 || nc<4 || bytes>=size_t(nc)*start*th) assert(nt==start);
  if(mode==2 && nc>=4 && bytes>=size_t(nc)*128*th) assert(bytes>=size_t(nc)*nt*th);
 }
 int nt=512;balanceSwitchlessThreads(2,512,4,1,&nt);assert(nt==128);
 nt=512;balanceSwitchlessThreads(2,40960,4,1,&nt);assert(nt==512);
}
"""
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cc').write_text(code)
 subprocess.run(['c++','-O1','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(p/'test.cc'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('PASS: actual adaptive thread helper preserves large-call blocks and inactive modes; retains four 128-thread blocks for the 128-byte/rank reduce-scatter case')
