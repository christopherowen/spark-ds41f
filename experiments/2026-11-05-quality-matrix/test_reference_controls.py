"""Recorded native controls cannot be mixed across runtime, input or boot conditions."""
import copy,datetime,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import reference_controls as refs

class RecordedControls(unittest.TestCase):
    def fixture(self,root,benchmark='longbench'):
        folder=root/'reference';folder.mkdir();(root/'runtime.py').write_text('current\n')
        source={'commit':'old','benchmark':benchmark,'split':'pilot','ruler_stage':'screen' if benchmark=='ruler' else 'full','nodes':['node']}
        (folder/'source.json').write_text(json.dumps(source));(folder/'restored.json').write_text(json.dumps({'restored':True,'holds_removed':True}))
        row={'id':'one','prompt_sha256':'input','correct':True,'predicted':'A','score':100.,'status':'complete','finish_reason':'stop'}
        if benchmark=='longbench':run={'complete':True,'selection_sha256':'selection','dataset_sha256':'data','protocol':{'temperature':0,'max_tokens':65536},'phases':[{'name':'cold','concurrency':1,'cache_reuse':False,'records':[row]}]}
        else:run={'complete':True,'manifest_sha256':'screen','full_manifest_sha256':'full','protocol':{'temperature':0,'max_tokens':65536},'split':'pilot','stage':'screen','records':[row]}
        day=datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        for label in ('native-a','native-b'):
            for name,data in [('answers-'+label+'.json',run),('answers-'+label+'-conditions.json',{'halted':[],'fan_stop_failures':[],'thermal':{'node':{'thermal_slowdown_ms':0}}}),('benchmark-'+label+'-exit.json',{'exit':0,'started_at':day+'T12:00:00+00:00'})]:
                (folder/name).write_text(json.dumps(data))
        return folder
    def load(self,root,benchmark='longbench',old=b'current\n'):
        def git(argv,**kwargs):
            if argv[1]=='show':return old
            self.assertEqual(argv[1:3],['cat-file','-e']);return b''
        with patch.object(refs,'dependency_paths',return_value=['runtime.py']),patch.object(refs.subprocess,'check_output',side_effect=git):
            return refs.load(root,Path('reference'),benchmark,'pilot','screen' if benchmark=='ruler' else 'full',['node'])
    def test_complete_pair_is_returned_byte_for_byte_with_provenance(self):
        for benchmark in ('longbench','ruler'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);reference=self.fixture(root,benchmark);runs,files,meta=self.load(root,benchmark)
                self.assertEqual(set(runs),{'native-a','native-b'});self.assertEqual(len(files),6)
                self.assertEqual(files['answers-native-a.json'],(reference/'answers-native-a.json').read_bytes())
                self.assertEqual(meta['source_commit'],'old');self.assertIn('not interleaved',meta['scope'])
    def test_changed_runtime_requires_fresh_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            with self.assertRaisesRegex(AssertionError,'fresh native controls'):self.load(root,old=b'different')
    def test_old_failed_or_incomplete_controls_are_rejected(self):
        for kind in ('date','exit','incomplete','thermal','restoration','different_answer','protocol','topology','split'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);reference=self.fixture(root)
                name={'date':'benchmark-native-a-exit.json','exit':'benchmark-native-a-exit.json','thermal':'answers-native-a-conditions.json','restoration':'restored.json','topology':'source.json','split':'source.json'}.get(kind,'answers-native-b.json')
                p=reference/name;data=json.loads(p.read_text())
                if kind=='date':data['started_at']='2000-01-01T12:00:00+00:00'
                elif kind=='exit':data['exit']=1
                elif kind=='incomplete':data['complete']=False
                elif kind=='thermal':data['thermal']['node']['thermal_slowdown_ms']=1
                elif kind=='restoration':data['holds_removed']=False
                elif kind=='different_answer':data['phases'][0]['records'][0]['predicted']='B'
                elif kind=='protocol':data['protocol']['max_tokens']=1
                elif kind=='topology':data['nodes']=['different']
                elif kind=='split':data['split']='heldout'
                p.write_text(json.dumps(data))
                with self.subTest(kind=kind),self.assertRaises(AssertionError):self.load(root)
    def test_unpublished_reference_and_unsupported_coding_reuse_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            with patch.object(refs.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,['git'])):
                with self.assertRaises(subprocess.CalledProcessError):refs.load(root,Path('reference'),'longbench','pilot','full',['node'])
            with self.assertRaises(AssertionError):refs.load(root,Path('reference'),'code','pilot','full',['node'])

if __name__=='__main__':unittest.main()
