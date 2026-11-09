Scripts as run on 2026-11-09 for the selection of candidates 51-53. The laptop only fetched,
listed and wrote receipts. Every patch application ran on sp8, in
~/spark-lab/agents/round-51-53-select.

Laptop, in a scratch directory with a --shared clone of the deployment repository (repo/):
  gh pr list --repo christopherowen/spark-ds41f --state open --limit 200 \
    --json number,title,body,files,headRefName,headRefOid,author,createdAt,updatedAt,baseRefName,isDraft,labels,mergeable > prs.json
  git -C repo fetch origin main $(for n in <pool and passed PRs>; do echo +refs/pull/$n/head:refs/review/pr-$n; done)
  python3 -I extract.py <scratch>     # v/<N>/<component>/*.patch, scan.json (targets, heads)
  (v/, scan.json, pool.txt and prs.json copied to sp8 with tar over ssh)

sp8, the base tree:
  git clone --no-hardlinks --no-checkout ~/spark-lab/agents/candidate-50-rework-1009/vllm-c50 base-vllm
  git -C base-vllm checkout -B round-51-53-base ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9   # tree d5778ab5
  GIT_AUTHOR_DATE=2026-11-09T00:00:00Z GIT_COMMITTER_DATE=2026-11-09T00:00:00Z git -C base-vllm revert --no-edit 0a231de28   # 0778, #1009
  -> 0bffb886ec85, tree 14c5a2dc9245. The four non-test files 0778 touches equal
     candidate-50/no1009/overlay by SHA-256.
  (The source tree is identical to dgx4's candidate-50 prepared tree: same HEAD and tree. It was
   copied on sp8, so the dgx ring was not loaded during its lab window.)

sp8, in order:
  python3 apply_alone.py                    # alone.json; rebased/<N>.diff for every PR that applies
  python3 show_conflicts.py <N...>          # conflict-regions.json: test append vs code conflict
  python3 rebase_context.py 953 954 967 971 987 1001 1003   # rebase-context.json, rebased/<N>.diff
  python3 derive_split.py 987 987-wob vllm/models/deepseek_v4_1/tilelang/gemm.py \
      vllm/models/deepseek_v4_1/tilelang/wo.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py
  python3 pairs.py 953 954 960 961 962 963 967 968 969 971 977 981 983 984 987 987-wob 988 992 995 998 1001 1002 1003 1006 1008 1014 1030
  bash other_upstreams.sh                   # TileLang 2123e412 + 1107/tilelang/series; sparknet 1fd6a1fb
  bash other_pairs.sh
  python3 stack.py '{"51":["961","963","987-wob","1001"],"52":["1030","968","969"],"53":["960","971","977","984","998"]}'

stack.py was run twice. On the first run, its reset moved the earlier candidates' scratch branches
back to the base, after their trees and patches had been recorded. The second run, with a checkout in
place of the reset, gave the same trees.

Laptop:
  python3 -I receipts.py <scratch> <this receipts directory>
