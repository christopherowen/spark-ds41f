# spark-ds41f-1107: release qualification records

**Keep these records with the release, for good.** They are what later rounds compare
against:
- the next candidates' references, so a released build is never measured again;
- the previous-release column of the next release notes;
- the only evidence of how 1107 behaved on each shape.

A record left only on a node is lost to the repository: it can't be found, reviewed or
compared. Every release commits all of its qualification records at promotion
([methodology](../../../../docs/methodology.md#promotion)).

**How these were recovered.** 1107's promotion committed its tp4-ring and tp3-triangle suite
runs to `manifests/suites`, but only `bin/spark bench` reports for the switched shapes. The
rest stayed on the nodes until 2026-11-09:
- the switched suite runs;
- the quality evals, AgentX and the gate;
- the repeat boots;
- startup memory, collective probes and telemetry.

Candidates 48 and 49 then booted 1107 as a middle arm, measuring a released build again,
because its switched references could not be found. All of it was copied here from the
nodes on 2026-11-09, unchanged.

## Where everything is

**Suite references.** These are under `manifests/suites/`, which is where the lab's
`"reference"` reads them:

| Shape | Release suite | Companions |
| --- | --- | --- |
| tp4-ring | `2026-11-07-spark-ds41f-1107-tp4` (committed at promotion) | `-sampled`, `-single-stream` |
| tp3-triangle | `2026-11-07-spark-ds41f-1107-tp3` (committed at promotion) | `-sampled`, `-single-stream` |
| tp4-switched | `2026-11-07-spark-ds41f-1107-tp4-switched` (sp1) | `-sampled`, `-single-stream` |
| tp3-switched | `2026-11-07-spark-ds41f-1107-tp3-switched` (sp5) | `-sampled`, `-single-stream` |
| tp8-switched | `2026-11-07-spark-ds41f-1107-tp8-switched` (sp1) | `-sampled`, `-single-stream` |

All of them match the current suite definitions (`suite_sha256`, checked 2026-11-09). As at
promotion, the model step's generated text (`model.json.gz`) is left out.

**This folder:**

| Folder | From | What |
| --- | --- | --- |
| `tp4-ring/` | dgx1 `~/spark-lab/1107-tp4-ring` | quality evals (`evals/`: LongBench v2, RULER, EvalPlus HumanEval, Vision Eval, tool-eval-bench), the gate (`gate/`, including the 960K needle), collective probes, telemetry, step scripts and logs |
| `tp4-ring/repeat-boot/` | dgx1 `results/private/suites/1107-release*/spark-ds41f-1107-again` | the repeat boot of the tp4-ring release, sampled and single-stream suites |
| `tp3-triangle/` | dgx1 `~/spark-lab/1107-tp3-triangle` | the gate, telemetry, step scripts and logs |
| `tp3-triangle/repeat-boot/` | dgx1 `results/private/suites/1107-release-tp3*/spark-ds41f-1107-again` | the repeat boot at tp3-triangle |
| `switched-sp1/` | sp1 `~/spark-lab/1107-switched` | tp4-switched (AgentX c8, two bench boots, a repeat boot, startup memory, collective probes, and 1030's comparison suite runs), tp5..tp8-switched bench and probe runs, `run_tp_shape.py` and its logs |
| `switched-sp5/` | sp5 `~/spark-lab/1107-switched` | tp3-switched bench boots, collective probes |
| `tp4-ring/run_evals_dgx4.sh`, `tp4-ring/agentx/` | dgx4 `~/spark-lab/1107-tp4-ring` | the eval runner that `run_step3.sh` calls over ssh (the five evals, then AgentX c8 against tp4-ring), and AgentX c8's results |
| `tp3-triangle/run_evals_dgx4.sh`, `tp3-triangle/evals/` | dgx4 `~/spark-lab/1107-tp3-triangle` | the eval runner that tp3-triangle's `run_step3.sh` calls, and its quality evals (LongBench v2, RULER and EvalPlus were run twice; the files are the retry's) |

The dgx4 files were copied on 2026-11-09, unchanged (SHA-256 checked against dgx4). The tp4-ring
evals copied from dgx1 earlier are byte-identical to dgx4's. tp3-triangle's
`evals/vision-eval/score.json` is empty on dgx4 too: its runner passed the data directory, not
`items.jsonl`, to `score.py`.

**What the eval runners call.** The harness scripts are committed and unchanged since
`79b5e11f4f`. dgx4's checkout ran them at `8b193039a4`. These inputs exist only on the nodes:

| Input | Where | Size |
| --- | --- | --- |
| tool-eval-bench `3.0.1.dev10+g1cd7309e5` (MIT, github.com/SeraphimSerapis/tool-eval-bench at `1cd7309e5bf0`) | dgx4 `~/spark-lab/tool-eval-bench-latest/` (`.venv`, `repo`) | 127 MB |
| LongBench v2 cache, revision `ac35867b`, policy `65536-v1` | dgx4 `~/spark-lab/datasets/longbench-v2/` (also sp1) | 24 MB |
| RULER cache, revision `db43f92f`, policy `65536-v1`, pilot | dgx4 `~/spark-lab/datasets/ruler/` (also sp1) | 82 MB |
| EvalPlus `v0.1.10` | dgx4 `~/spark-lab/datasets/evalplus/` (also sp1) | 2.9 MB |
| Vision Eval data (2,283 items) | dgx4 and dgx1 `~/spark-lab/vision-eval/data` | 630 MB |

The caches are built by the committed preparation steps of
[answer-quality](../../../2026-11-05-answer-quality/README.md) (`prepare.py`,
`code_quality.py prepare`) and [context-quality](../../../2026-11-05-context-quality/README.md)
(`prepare.py`), and the Vision Eval data by
[vision-eval](../../../2026-10-25-vision-eval/README.md)'s `fetch.sh` and `extract.py`.

**Left on the nodes.** These are in `large-files-on-nodes.tsv`, with host, path, size and
SHA-256:
- files over 5 MB: AgentX's raw harness exports;
- the model step's generated text;
- `evals/private`: the raw LongBench, RULER and HumanEval+ items behind the scores (on dgx4,
  203 files for each of tp4-ring and tp3-triangle, not listed one by one).
