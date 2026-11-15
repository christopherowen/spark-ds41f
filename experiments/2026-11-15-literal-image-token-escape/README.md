# The literal image token in text: escape it, for the candidate-55 base

Rebase of [#1047](https://github.com/christopherowen/spark-ds41f/pull/1047) (vLLM `0762`) onto the
candidate-55 patch head, carried here as **vLLM `0811`**. Fixes [#1031](https://github.com/christopherowen/spark-ds41f/issues/1031).

## Why it needed rebasing

`0762` was written against the 1107 patch head (`2756478b`). It does not apply to candidate-55's tree:
candidate-55 rewrote `_normalize_messages` and `deepseek_v41_encoding`, so three of its four hunks apply
only with context fuzz and the fourth — the escaping helper — fails outright.

```
$ git apply --check 0762-fix-dsv41-escape-the-literal-image-token-in-text.patch
error: patch failed: vllm/tokenizers/deepseek_v41.py:14
error: vllm/tokenizers/deepseek_v41.py: patch does not apply
```

So candidate-55 (build 9) shipped **without** the fix. We confirmed that by behaviour on a live TP3
ring, not by reading the series: the literal token gets HTTP 400 in candidate-55, and 200 in our
candidate-36-0762 image, which carries `0762`.

## What this patch does

Identical behaviour to `0762`, adapted to candidate-55's code shape:

- `_escape_image_token_literal(value)` replaces `IMAGE_PLACEHOLDER` (the token's fullwidth-bar spelling)
  with `IMAGE_PLACEHOLDER_TEXT` (the ASCII-bar ordinary-text spelling, in no added vocabulary) across
  strings, lists and dicts of a message.
- `_normalize_messages` calls it on every message first, so the rejecting branches it replaced can no
  longer see the token in request text. Real image parts still render as `IMAGE_PLACEHOLDER`.
- candidate-55's `_normalize_messages` uses `[dict(message) for message in messages]` rather than
  `copy.deepcopy(messages)`; the call is inserted after that line accordingly.

**Source only.** #1047's test changes are not carried: they need rebasing against candidate-55's
`tests/tokenizers_/test_deepseek_v41.py`, and we have no way to run your release-test bundle. A
behavioural receipt is included instead (below).

## Evidence

On a third-party ring, 3× GB10 switchless triangle (`tp3-triangle`), image built from the candidate-55
lock with this patch as `0811`:

| build | request text contains the literal `<|deepseek_image|>` |
| --- | --- |
| `candidate-55` build 9, tree `49493238c2ff` | **400** `Message content contains image special token '<|deepseek_image|>'` |
| candidate-55 + `0811`, tree `c34712803211` | **200**, normal completion |
| `candidate-36-0762` (carries `0762`) | **200**, normal completion |

The patch applies cleanly to the candidate-55 tree (`git apply --check`, hunks 1-5, no fuzz), the tree
moves `49493238c2ff` → `c34712803211`, and the ring booted and served with it (155 s warm boot, the
recipe's `post-update-check` 8/8). Rebased patch, receipts and the exact commands are under `receipts/`.

## What it does not prove

- The release-test bundle has not run against it (no kernel-lab access here), so this is not a
  release-qualified patch — it is a rebase plus a behavioural check on one ring.
- No throughput measurement on the patched image; `0811` touches request normalisation only, not the
  forward pass, so we do not expect one to move, but that is reasoning rather than a measurement.
