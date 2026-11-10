"""Exercise the actual routing method without a model or GPU reference work."""
import ast,os,types,unittest
from pathlib import Path
from unittest.mock import patch

class Native:
    def scratch_specs(self,rows):return ()
    def run(self,*args,**kwargs):self.called=True
class Sparse(Native):
    budget=64;page=128

class Dispatch(unittest.TestCase):
    def test_modes_existing_native_paths_and_covered_fallback(self):
        tree=ast.parse(Path(__file__).with_name('overlay').joinpath('attention.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41TileLangAttention')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_index_chunk')
        workspace=types.SimpleNamespace(get_simultaneous=lambda *args:())
        scope={'os':os,'PageShortlistPlan':Sparse,'current_workspace_manager':lambda:workspace,'retain_cuda_graph_capture_resource':lambda *args:None}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'attention routing','exec'),scope)
        for flag,mode,width,want_native in [('1','decode',None,True),('1','prefill',None,False),('1','prefill_short',None,False),('0','decode',None,False),('0','prefill',1000,True),('1','prefill',1000,True)]:
            for rows in (1,8,64):
                approximate=Sparse();native=Native()
                plans=types.SimpleNamespace(index={mode:approximate},native_index={mode:native})
                layer=types.SimpleNamespace(_tilelang=lambda:plans,indexer=types.SimpleNamespace(k_cache=types.SimpleNamespace(kv_cache=None)),_active=None)
                with patch.dict(os.environ,{'VLLM_DS41_PAGE_NATIVE_DECODE':flag}):
                    scope['_index_chunk'](layer,mode,rows,output_indices=None,q_mxfp4=None,q_scales=None,query_weights=None,page_table=None,cache_lengths=None,score_width=width)
                self.assertEqual(getattr(native,'called',False),want_native)
                self.assertEqual(getattr(approximate,'called',False),not want_native)
        # Source/reindex plans that are native already have no fallback entry.
        native=Native();plans=types.SimpleNamespace(index={'decode':native},native_index={})
        layer=types.SimpleNamespace(_tilelang=lambda:plans,indexer=types.SimpleNamespace(k_cache=types.SimpleNamespace(kv_cache=None)),_active=None)
        with patch.dict(os.environ,{'VLLM_DS41_PAGE_NATIVE_DECODE':'1'}):
            scope['_index_chunk'](layer,'decode',8,output_indices=None,q_mxfp4=None,q_scales=None,query_weights=None,page_table=None,cache_lengths=None)
        self.assertTrue(native.called)

if __name__=='__main__':unittest.main()
