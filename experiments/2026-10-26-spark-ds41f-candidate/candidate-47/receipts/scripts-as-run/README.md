Scripts as run on 2026-11-09 from a scratch directory, kept for review. They read
prs.json (gh pr list --json number,title,body,files,...) and refs/review/pr-<N>
(git fetch origin pull/<N>/head), and apply patches to a scratch vLLM clone at
1107 (base 19f2c20ed + 1107/vllm/series, tree ed1bdb7a).
