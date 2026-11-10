Scripts as run on 2026-11-10 for the selection of candidates 55-57. The laptop only fetched, listed and
wrote receipts. Every patch application ran on sp8's CPU, in ~/spark-lab/agents/round-55-57-select. sp8's
GPU was in use by another agent and was not touched.

Laptop, in a scratch directory with a fresh clone of the deployment repository (repo/, origin/main 920e1d50):
  gh pr list --repo christopherowen/spark-ds41f --state open --limit 200 \
    --json number,title,body,files,headRefName,headRefOid,author,createdAt,updatedAt,baseRefName,isDraft,labels,mergeable > prs.json
  for n in <pool> 1047 1005; do gh pr view $n --json number,author,headRefOid,headRefName,updatedAt,commits; done   # outputs/authors.tsv
  git -C repo fetch origin $(for n in <pool> 1047 1005 984; do echo +refs/pull/$n/head:refs/review/pr-$n; done)
  python3 -I extract.py <scratch>     # v/<N>/<component>/*.patch, scan.json (unchanged from candidate-51's)
  (candidate-54's 671 series patches, resolved to repository paths in c54-series.txt, and v/, scan.json,
   pool.txt and prs.json copied to sp8 with tar over ssh)

sp8, in order:
  bash build_base.sh                        # c54-vllm at 4e8133ff, tree 5f021d9a (candidate-54's lock); branch round-55-57-base
  python3 apply_alone.py                    # outputs/alone.json; rebased/<N>.diff for the PRs that apply
  python3 show_conflicts.py 953 956 967 978 995 1003 1006   # outputs/conflict-regions.json (two-way markers)
  git -C c54-vllm config merge.conflictstyle diff3
  python3 split_hunks.py 988 988-deadrow '{...}'   # and 988-topk, 957-p1..p5, 957-p5code, 995-alloc-apply, 995-match:
                                                   # the specs are in outputs/splits.json ("kept")
  python3 rebase_context.py <N> v/<N>/vllm/<patch>  for 967 978 953 956 995 1003 1006
  python3 rebase_context.py 957-p5 splits/957-p5.patch; python3 rebase_context.py 957-p3 splits/957-p3.patch
                                            # outputs/rebase-context.json: context-only (967, 978) or semantic
  python3 pairs.py 967 978 988-deadrow 957-p1 957-p2 957-p4 1008   # outputs/pairs.json
  bash other_upstreams.sh                   # outputs/other-upstreams.txt (TileLang 0014 of #967, sparknet 0001 of #1008)
  python3 stack.py '{"55":["1008","978","988-deadrow"],"56":["957-p1"],"57":["957-p2","957-p4"]}'
                                            # out/<cand>/, outputs/stack.json, ../stack.log; union tree 85ae9bf1
  python3 - < base_shared.py               # outputs/base-shared-files.json (after files.json was copied over)

Laptop, between the sp8 steps:
  python3 -I - < overlaps.py               # outputs/files.json: every set's files and every overlapping pair

stack.py ran three times. The first two runs used an earlier order for 55 (978 before 1008) and an earlier output
naming. The trees were the same each time, and the last run (the final order) is the one recorded.

Laptop:
  python3 -I receipts.py <scratch> <this receipts directory>     # pr-patch-targets.json, pairwise-conflicts.json, rebases/
  python3 -I selection.py <scratch> <this receipts directory>    # selection-proposal.json
