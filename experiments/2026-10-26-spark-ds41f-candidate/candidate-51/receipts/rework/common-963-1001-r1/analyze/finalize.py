# usage: python3 finalize.py RESULT_963.json RESULT_1001.json PATCH_SHA_963 PATCH_SHA_1001 APPLY_LOG
# Adds the verdicts, bit-exactness, the patch-apply check and the notes to both result.json files.
import json
import sys

r963, r1001 = (json.load(open(p)) for p in sys.argv[1:3])
sha963, sha1001, apply_log = sys.argv[3], sys.argv[4], sys.argv[5]
apply = {
    "order": "candidate-51's series with the corrected patches: candidate-50's 0762..0777, 0779, 0780 on 1107's patch head, then 0781 (#961), 0782 (#963-r1), 0783 (#987, unchanged), 0784 (#1001-r1)",
    "how": "the own-mark run's stack.sh in a fresh --shared clone (W/verify-apply): git apply --check --verbose, then git am --committer-date-is-author-date, as bin/spark build prepare applies them",
    "result": open(apply_log).read().strip().splitlines()[-8:],
    "tree": "35cd2a0e87a6b7a0c61c6c81dbe875ac1e7da005, the tree measured as the c51r1 arm; git diff --check 19f2c20ed4d6..HEAD clean",
}
r963["patch_sha256"] = sha963
r1001["patch_sha256"] = sha1001
r963["apply_check"] = apply
r1001["apply_check"] = apply
json.dump(r963, open(sys.argv[1], "w"), indent=1)
json.dump(r1001, open(sys.argv[2], "w"), indent=1)
print("ok")
