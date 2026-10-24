# Prepared source recipe

The ordered `vllm/series` starts at upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. `identity.json` records the full source
head, tree, replay head and replay tree; `replay.txt` retains all 274 successful
applications. The replay and source trees are identical. Original license
notices remain in the exports.

The control changes start after source base `47176036f`; earlier patches supply
the combined ports already present at that base. This experiment does not
requalify those earlier patches or select this entire recipe for deployment.
The complete stack is retained so the new candidates have a reproducible source
context rather than depending on a mutable local checkout.

The control patches are local experimental work, not submitted upstream. Their
commit messages explain each translation or screening fix. Their cumulative
purpose is to preserve the existing public arithmetic and state boundaries in
TileLang while testing whether that translation can avoid a performance loss.
No control dispatch is switched. Required public-owner startup registration,
full-model validation and serving measurements remain pending.

The final control qualification uses the tests and benchmark identified in
`../qualified/source.json`; exact commands, container digest, bundle hashes,
native output, failures and limitations are retained alongside it and in the
parent README. Passing the 141-case suite and 50-case memcheck does not qualify
the other patches in this stack or establish end-to-end output equality.

To replay in a fresh checkout at the pinned upstream commit:

```sh
while IFS= read -r patch; do
    git am "/absolute/path/to/prepared-source/vllm/$patch" || exit 1
done < /absolute/path/to/prepared-source/vllm/series
git rev-parse HEAD^{tree}
```

Expected tree: `561b6faa709ec1274de1726e4e17a4a4436e76ef`.
