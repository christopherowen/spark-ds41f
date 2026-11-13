#!/bin/bash
# Build revert trees of candidate-54 (4e8133ff, tree 5f021d9a) as git worktrees.
set -euo pipefail
W=~/spark-lab/agents/tp3-grid-attribution
cd $W/c54
git -c advice.detachedHead=false checkout -q 4e8133ff10a125b90753c2727f162e29fe37f30c
declare -A C=(
  [961]=97f84b929 [963]=5d458ff23 [987]=056cc4763 [1001]=20a20630d
  [1030]=07b80ffd2 [968]=1475da08c [969]=f297875de [960]=972053790 [998]=4e8133ff1
)
mk() {  # name, PRs...
  local name=$1; shift
  local dir=$W/trees/$name
  if [ -d $dir ]; then git worktree remove --force $dir; fi
  git worktree add -q --detach $dir 4e8133ff10a125b90753c2727f162e29fe37f30c
  ( cd $dir
    git config user.name "tp3-grid-attribution"; git config user.email "none@localhost"
    # Revert newest first so later patches on shared files unwind in order.
    local commits=()
    for pr in "$@"; do commits+=("$(git rev-parse ${C[$pr]})"); done
    local ordered=$(git rev-list --no-walk=sorted "${commits[@]}")
    for c in $ordered; do git revert --no-edit $c >/dev/null; done
    echo "$name: head $(git rev-parse --short=12 HEAD) tree $(git rev-parse HEAD^{tree}) reverted: $*" )
}
mk no961 961
mk no963 963
mk no987 987
mk no1001 1001
mk no1030 1030
mk no969 969
mk no960 960
mk no998 998
mk no968 968
mk noC51 961 963 987 1001
mk noC52 1030 968 969
mk noAll 961 963 987 1001 1030 968 969 960 998
