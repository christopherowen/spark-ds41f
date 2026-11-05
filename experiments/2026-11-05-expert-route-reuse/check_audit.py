"""CPU compiler/geometry audit in the pinned image; no GPU or device compile."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import tilelang
from tilelang import tvm

from vllm.models.deepseek_v4_1.tilelang import moe as moe_module
from vllm.models.deepseek_v4_1.tilelang.moe import (
    MoeExpertsV41Plan, grouped_fp8_fp4_gemm, workspace_bytes,
)
from analyze_ordering import PANELS, cta_order


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def check_header():
    header = Path(tilelang.__file__).parent / 'src/tl_templates/cuda/threadblock_swizzle.h'
    original = header.read_text()
    stub = '''#include <cstdint>
#define TL_DEVICE inline
struct dim3 { unsigned int x,y,z; dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
static dim3 blockIdx, gridDim;
'''
    source = original.replace('#include "common.h"', stub)
    assert source != original
    bridge = '''
extern "C" void rasterize(int p,int nx,int ny,int x,int y,unsigned int* out) {
 gridDim=dim3(nx,ny,1); blockIdx=dim3(x,y,0); dim3 value;
 switch(p) {
 case 0: value=blockIdx; break;
'''
    bridge += ''.join(f' case {p}: value=tl::rasterization2DRow<{p}>(); break;\n' for p in PANELS if p)
    bridge += ' default: value=dim3(99999,99999,99999);\n }\n out[0]=value.x;out[1]=value.y;out[2]=value.z;\n}\n'
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cpp=root/'bridge.cpp';lib=root/'bridge.so'
        cpp.write_text(source + bridge)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
        dll=ctypes.CDLL(str(lib));fn=dll.rasterize
        fn.argtypes=[ctypes.c_int]*5+[ctypes.POINTER(ctypes.c_uint)]
        fn.restype=None;out=(ctypes.c_uint*3)()
        for nx in (1,2,3,9,12,40):
            for ny in (*range(1,18),31,32,33,63,64,65,384,385,1152):
                for panel in PANELS:
                    seen=set()
                    for linear, expected in enumerate(cta_order(nx,ny,panel)):
                        fn(panel,nx,ny,linear%nx,linear//nx,out)
                        actual=(out[0],out[1]);assert actual==expected,(nx,ny,panel,linear,actual,expected)
                        assert out[2]==0
                        seen.add(actual);checked+=1
                    assert len(seen)==nx*ny
    return {'header_sha256':digest(original),'coordinates_checked':checked,'result':'pass'}


def check_compiler():
    target=tvm.target.Target({'kind':'cuda','arch':'sm_121'})
    results=[]
    for gather,n,k in ((True,1152,5120),(True,1536,5120),
                       (False,5120,576),(False,5120,768)):
        for m in (16,32,64):
            plain=None
            for panel in PANELS:
                stages=2 if not gather and m<64 else 3
                func=grouped_fp8_fp4_gemm.get_tir(
                    384,n,k,6,gather,m,128,num_stages=stages,
                    coded_scales=True,narrow_codes=True,
                    trigger_after_loop=True,swizzle_panel=panel)
                with target,tvm.transform.PassContext(opt_level=3,config=grouped_fp8_fp4_gemm.pass_configs):
                    artifact=tilelang.lower(func,target=target,
                                           enable_device_compile=False,
                                           enable_host_codegen=False)
                source=artifact.kernel_source
                if panel==0:
                    plain=source
                    assert 'rasterization2DRow<' not in source
                else:
                    marker=f'  const dim3 blockIdx = tl::rasterization2DRow<{panel}>();\n'
                    assert source.count(marker)==1,(gather,n,k,m,panel)
                    assert not re.search(r'\bblockIdx\b', source.split(marker)[0])
                    assert source.replace(marker,'')==plain,(gather,n,k,m,panel,'other emitted code changed')
                results.append({'gather':gather,'n':n,'k':k,'m':m,'panel':panel,
                                'stages':stages,'source_sha256':digest(source),
                                'body_without_remap_sha256':digest(plain)})
    return results


def check_default():
    target=tvm.target.Target({'kind':'cuda','arch':'sm_121'})
    checked=[]
    for inter in (576,768):
        for stage in ('gate_up','down'):
            sources={}
            for panel in (None,0,3):
                options={} if panel is None else {'prefill_swizzle_panel':panel}
                plan=MoeExpertsV41Plan(384,6,5120,inter,**options)
                factory,args,kwargs=plan.kernels()[f'prefill_{stage}']
                func=factory.get_tir(*args,**kwargs)
                with target,tvm.transform.PassContext(opt_level=3,config=factory.pass_configs):
                    source=tilelang.lower(func,target=target,enable_device_compile=False).kernel_source
                assert ('rasterization2DRow<3>' in source)==(panel==3)
                sources[panel]=source
            assert sources[None]==sources[0]
            checked.append({'intermediate':inter,'stage':stage,
                            'default_source_sha256':digest(sources[None]),
                            'explicit_3_source_sha256':digest(sources[3])})
    return checked


def check_workspace():
    results=[]
    for tp in (3,4):
        inter=2304//tp
        for m in (16,32,64):
            plan=MoeExpertsV41Plan(384,6,5120,inter,
                                  decode_block_M=m,block_M=m,prefill_block_M=m)
            for rows in (1,17,48,1023,1024,2047,2048,8192):
                specs=plan.workspace_specs(rows)
                allocated=min(rows*6,(rows*6+m-1)//m+384)
                scratch=allocated*m*(5*inter+4*((inter+127)//128)+2*5120)
                # Last four buffers are gate/up output, down input/scales and down output.
                exact=sum(__import__('math').prod(shape)*dtype.itemsize for shape,dtype in specs[-4:])
                assert exact==scratch
                results.append({'tp':tp,'m':m,'rows':rows,'gemm_intermediates_bytes':exact,
                                'full_aligned_workspace_bytes':workspace_bytes(specs)})
    return results


if __name__=='__main__':
    result={'native_device_compilation':False,'gpu_execution':False,
            'runtime_source_sha256':digest(Path(moe_module.__file__).read_text()),
            'rasterization':check_header(),'compiler_cases':check_compiler(),
            'default_plan_cases':check_default(),
            'workspace_cases':check_workspace(),'result':'pass'}
    print('RESULT_JSON\n'+json.dumps(result,indent=2))
