import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path('/home/chris/projects/engram-cache-probe')
repo = root / 'deployment'
relative = Path('experiments/2026-11-05-engram-cache-probe')
dest = repo / relative
shutil.copyfile(root / 'collect.py', dest / 'runs/inputs/collect.py')
shutil.copyfile(root / 'finalize.py', dest / 'runs/inputs/finalize.py')

def git(*args):
    return subprocess.check_output(['git', '-c', 'index.threads=1', '-c', 'core.preloadIndex=false', *args], cwd=repo, text=True).strip()

base_commit = '2c4b23afcd834ba2adf6a15ca434e59fa4f49f74'
amend = git('rev-parse', 'HEAD') != base_commit
if amend:
    assert git('rev-parse', 'HEAD^') == base_commit
    assert git('branch', '--show-current') == 'feat/engram-cache-probe-20261105'
else:
    git('switch', '-c', 'feat/engram-cache-probe-20261105')
assert all(x.startswith(str(relative) + '/') for x in git('diff', '--name-only').splitlines())
assert all(x.startswith(str(relative) + '/') for x in git('diff', '--cached', '--name-only').splitlines())
git('add', str(relative))
for name in ('doctor-initial.log', 'doctor.log', 'doctor-tp3.log', 'doctor-tp4.log',
             'prepare-retry.log', 'source-patch.log', 'lock.log'):
    git('add', '-f', str(relative / 'runs/setup' / name))
git('diff', '--cached', '--check')
files = git('ls-files', '--', str(relative)).splitlines()
assert files and all(x.startswith(str(relative) + '/') for x in files)
assert all(not x.endswith(('/base', '/candidate', '.pyc')) for x in files)
message = 'Record bounded Engram cache-probe experiment\n\nSeparate cache probing from admission for direct steps through 128 rows.\nKeep the candidate-38 base and one source variable in an isolated patch stack.\nRetain both final CPU trials, the rejected broad trial and all setup errors.\n\nThe native planner byte oracle and ASan/UBSan fixtures pass; fresh patch\napplication and static configuration checks pass. Planned read reductions\nare synthetic. GPU regressions, physical I/O and two serving boots remain\npending; no production configuration or immutable evidence is changed.\n'
(root / 'deployment-commit-message.txt').write_text(message)
git('-c', 'user.name=Christopher Owen', '-c', 'user.email=3221756+christopherowen@users.noreply.github.com',
    '-c', 'commit.gpgsign=false', 'commit', *(['--amend'] if amend else []), '-F', str(root / 'deployment-commit-message.txt'))
head = git('rev-parse', 'HEAD')
tree = git('rev-parse', 'HEAD^{tree}')
parent = git('rev-parse', 'HEAD^')
data = {'base_tree': git('rev-parse', parent + '^{tree}'),
        'tree': [{'path': f, 'mode': git('ls-files', '-s', '--', f).split()[0],
                  'type': 'blob', 'content': (repo / f).read_text()} for f in files]}
(root / 'github-tree.json').write_text(json.dumps(data))
payload = {'message': message, 'tree': tree, 'parents': [parent],
           'author': {'name': git('show', '-s', '--format=%an'), 'email': git('show', '-s', '--format=%ae'), 'date': git('show', '-s', '--format=%aI')},
           'committer': {'name': git('show', '-s', '--format=%cn'), 'email': git('show', '-s', '--format=%ce'), 'date': git('show', '-s', '--format=%cI')}}
(root / 'github-commit.json').write_text(json.dumps(payload))
(root / 'deployment-identity.json').write_text(json.dumps({'commit': head, 'tree': tree, 'parent': parent, 'files': len(files), 'branch': git('branch', '--show-current')}, indent=2) + '\n')
print((root / 'deployment-identity.json').read_text())
assert not git('status', '--porcelain')
