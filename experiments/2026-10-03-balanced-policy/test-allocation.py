"""Exercise NCCL's actual channel partition code with the proposed cell floors."""
from pathlib import Path
import importlib.machinery,importlib.util,argparse,subprocess,tempfile,os
ROOT=Path(__file__).resolve().parents[2]
l=importlib.machinery.SourceFileLoader('allocation_build',str(ROOT/'bin/spark'));s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
_,_,lock=m.configuration(argparse.Namespace(cluster_config=os.environ.get('BALANCED_TEST_PROFILE','experiments/2026-10-03-balanced-policy/paired.json')))
inputs=m.build_inputs(lock);tree=m.build_directory(inputs)/'src/nccl'
assert not m.project_problems(tree,inputs['nccl'])
source=(tree/'src/enqueue.cc').read_text()
a=source.index('      int trafficPerByte = ncclFuncTrafficPerByte');b=source.index('      // Update number of channels propagated',a)
partition=source[a:b]
code=r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
using std::size_t;
template<class T,class U> T divUp(T a,U b){return (a+b-1)/b;}
constexpr int NCCL_PROTO_LL=0;
int ncclFuncTrafficPerByte(int func,int ranks){return func==0?2:ranks;}
struct Comm{int nRanks;};
struct Task{int func,protocol;size_t count;};
int main(){
 for(int ranks:{3,4}) for(size_t elementSize:{2,4}) for(int func:{0,1,2})
 for(int proto:{0,1}) for(size_t MinTrafficPerChannel:{16,256,512,4096,32768})
 for(int channels:{1,2,3,4,8,16}) {
  Comm c{ranks};Comm* comm=&c;int nMaxChannels[1]={channels};int kind=0;
  for(size_t count=1;count<=8192;count++) {
   Task t{func,proto,count};Task* task=&t;
   size_t traffic=std::max(MinTrafficPerChannel,count*elementSize*size_t(ncclFuncTrafficPerByte(func,ranks)));
   size_t trafficPerChannel=divUp(traffic/size_t(channels),size_t(16))*16;
   if(!trafficPerChannel)continue;
   size_t currentTraffic=0;int channelId=0;
"""+partition+r"""
   assert(trafficPerElement>0);
   assert(countLo>0 && nChannels>=1 && nChannels<=channels);
   assert(nMidChannels>=0);
   assert(countLo+size_t(nMidChannels)*countMid+countHi==count);
   assert(channelId>=0 && channelId+nChannels<=channels);
   if(nMidChannels)assert(countMid>0);
   if(nChannels>1)assert(countHi>0);
  }
 }
}
"""
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cc').write_text(code)
 subprocess.run(['c++','-O1','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(p/'test.cc'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('PASS: extracted NCCL partition code preserves all elements without overlap or empty active channels over 1–8192 elements, 3/4 ranks, BF16/FP32, LL/Simple, 1/2/3/4/8/16 channels and 16–32768-byte floors')
