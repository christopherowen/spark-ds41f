"""Only exact owned evaluator processes are eligible for a fail-fast signal."""
from pathlib import Path
import tempfile
import unittest
from watch_quality import client_pids

class ClientIdentity(unittest.TestCase):
    def test_exact_parent_output_and_script_identity(self):
        root=Path('/controller');out=Path('/results/answers-lsa-a.json')
        with tempfile.TemporaryDirectory() as folder:
            proc=Path(folder)
            def add(pid,parent,script,target=out):
                p=proc/str(pid);p.mkdir()
                (p/'cmdline').write_bytes(b'\0'.join(str(a).encode() for a in ['python3',script,'--out',target])+b'\0')
                (p/'stat').write_text(f'{pid} (process with spaces) S {parent} 0 0')
            self.assertEqual(client_pids(root,out,proc),[])
            add(100,1,root/'experiments/2026-11-05-quality-matrix/measure.py')
            add(101,100,root/'experiments/2026-11-05-context-quality/evaluate.py')
            add(102,100,root/'experiments/2026-11-05-context-quality/evaluate.py',Path('/results/other.json'))
            add(103,1,root/'experiments/2026-11-05-context-quality/evaluate.py')
            add(104,100,Path('/foreign/experiments/2026-11-05-context-quality/evaluate.py'))
            self.assertEqual(client_pids(root,out,proc),[101])
            add(105,100,root/'experiments/2026-11-05-answer-quality/evaluate.py')
            self.assertEqual(client_pids(root,out,proc),[101,105])

if __name__=='__main__':unittest.main()
