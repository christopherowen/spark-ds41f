"""Legitimate repetition and loop shapes at real token counts, replayed through
check_stop as a visible answer (replay.py's replay), with the vLLM this runs on.

Kept shapes are legitimate output and should never stop; loop shapes follow 1,500
tokens of prose and should stop, the sooner the better. Prints each shape's token
count, then one JSON line: where each kept shape was cut (none is right), and how
many tokens after its start each loop stopped (null: never).

Run it like replay.py, in the serving image with the changed files mounted:
  python3 shapes.py TOKENIZER_JSON
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tokenizers import Tokenizer

from replay import replay

tok = Tokenizer.from_file(sys.argv[1])
enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
prefix = enc(open("/opt/spark-ds41f/candidate/vllm/README.md").read())[:1500]

z = lambda n: "```c\nstatic const int z[%d] = {%s};\n```" % (n, ", ".join(["0"] * n))  # noqa: E731
mat = lambda n, v: ("```python\nm = [\n" + "\n".join(  # noqa: E731
    "    [" + ", ".join([v] * n) + "]," for _ in range(n)) + "\n]\n```")
kept = {
    "hello-200": enc(" ".join(["hello"] * 200)),
    "zero-128": enc(z(128)),
    "zero-256": enc(z(256)),
    "ones-16x16": enc(mat(16, "1")),
    "sep-200": enc("Report\n\n" + "= " * 200 + "\n| p | p^2 |\n"
                   + "\n".join(f"| {p} | {p * p} |" for p in [2, 3, 5, 7, 11, 13]) + "\n" + "= " * 200),
    "csv-300": enc("a,b,c,d\n" + "0,0,0,0\n" * 300),
    "table-200": enc("| name | value |\n|---|---|\n" + "".join(f"| item{i} | 0 |\n" for i in range(200))),
    "zero-512": enc(z(512)),
    "zeros-32x32": enc(mat(32, "0.0")),
}
vila = enc(" Vila")[0]
pool = [enc(w)[0] for w in (" Evelyn", " Tripadvisor", " Escola", " Fest", " Estud", " compar", "的", "是", " Lisboa")]
rng = random.Random(7)
soup = []
while len(soup) < 4000:
    if rng.random() < 0.62:
        soup += [vila] * rng.randint(3, 12)
    else:
        soup += [rng.choice(pool) for _ in range(rng.randint(1, 3))]
loops = {
    "vila-run": [vila] * 4000,
    "vila-soup": soup[:4000],
    "let-me-do-it": enc("Let me do it. " * 1000)[:4000],
    "hello-runaway": enc(" ".join(["hello"] * 4000))[:4000],
}


def stop_at(ids):
    result = replay([], ids)
    return None if result is None else result[1]


print("tokens:", {name: len(ids) for name, ids in kept.items()})
cut = {name: stop_at(ids) for name, ids in kept.items()}
caught = {name: stop_at(prefix + ids) for name, ids in loops.items()}
print(json.dumps({"kept_cut": {name: at for name, at in cut.items() if at is not None},
                  "loop_stop_after": {name: None if at is None else at - len(prefix)
                                      for name, at in caught.items()}}))
