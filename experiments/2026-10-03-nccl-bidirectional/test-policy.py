from pathlib import Path
import subprocess,tempfile
import importlib.machinery,importlib.util,argparse
ROOT=Path(__file__).resolve().parents[2]
loader=importlib.machinery.SourceFileLoader("spark3_nccl_rings",str(ROOT/"bin/spark3"))
spec=importlib.util.spec_from_loader(loader.name,loader)
spark3=importlib.util.module_from_spec(spec);loader.exec_module(spark3)
_,_,lock=spark3.configuration(argparse.Namespace(cluster_config="experiments/2026-10-03-nccl-bidirectional/cluster.json"))
inputs=spark3.build_inputs(lock)
prepared=spark3.build_directory(inputs)/"src/nccl"
errors=spark3.project_problems(prepared,inputs["nccl"])
if errors:raise SystemExit("\n".join(errors))
source=(prepared/"src/graph/connect.cc").read_text()
a=source.index('static ncclResult_t balanceSwitchlessRings(')
b=source.index('\n}\n',a)+3
code=r'''
#include <cassert>
#include <vector>
#define WARN(...) ((void)0)
using ncclResult_t=int;
constexpr int ncclSuccess=0,ncclInvalidUsage=1;
struct Ring {int prev,next;};
struct Channel {Ring ring;};
struct Shared {struct ncclComm* owner;};
struct ncclComm {int nRanks,nChannels,nNodes,rank;Shared* sharedRes;Channel channels[16];};
'''+source[a:b]+r'''
int main() {
 for (int n: {3,4}) for (int channels: {2,4,8,16}) for (int rank=0;rank<n;rank++) {
  ncclComm c{};Shared s{&c};c={n,channels,n,rank,&s,{}};
  std::vector<int> p(n*channels),q(n*channels);
  auto init=[&](){for(int ch=0;ch<channels;ch++) {
   for(int r=0;r<n;r++){p[ch*n+r]=(r+n-1)%n;q[ch*n+r]=(r+1)%n;}
   c.channels[ch].ring={p[ch*n+rank],q[ch*n+rank]};
  }};
  init();assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclSuccess);
  for(int ch=0;ch<channels;ch++) {
   int direction=ch<channels/2?1:-1;
   for(int r=0;r<n;r++) {
    assert(q[ch*n+r]==(r+n+direction)%n);assert(p[ch*n+r]==(r+n-direction)%n);
    assert(p[ch*n+q[ch*n+r]]==r);
    int end=r;for(int hop=0;hop<n;hop++) end=q[ch*n+end];assert(end==r);
   }
   assert(c.channels[ch].ring.prev==p[ch*n+rank]);assert(c.channels[ch].ring.next==q[ch*n+rank]);
  }
  init();p.back()=-1;auto before=p;auto after=q;
  assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclInvalidUsage);assert(p==before&&q==after);
  init();c.nNodes--;assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclInvalidUsage);c.nNodes++;
  c.nChannels=1;assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclInvalidUsage);c.nChannels=3;
  assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclInvalidUsage);c.nChannels=channels;
  s.owner=nullptr;assert(balanceSwitchlessRings(&c,p.data(),q.data())==ncclInvalidUsage);
 }
 ncclComm c{};Shared s{&c};c={2,4,2,0,&s,{}};assert(balanceSwitchlessRings(&c,nullptr,nullptr)==ncclInvalidUsage);
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cc').write_text(code)
 subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(p/'test.cc'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('PASS: actual NCCL ring policy; 3/4 ranks, 2/4/8/16 channels, all rank views, adjacency, reciprocal rings, atomic rejection of unsupported geometry')
