# Existing-model barrier draft

No test, replay, build, reference-model read or serving operation ran while this
draft was written. Root must run the focused CPU tests remotely and review the
frozen C46 contract before enabling it. The template's candidate commit/image
remain null until the actual published execution and qualified image are known.

The opt-in touches `servebench/suite.py`, `servebench/__main__.py` and the focused
`servebench/model_evidence.py`; it does not modify lab.py, global runtime/build
inputs or any built-in suite definition. `tests/test_model_evidence.py` uses only
small fake model artifacts and fake step callbacks, with no HTTP or GPU request.

The C46 performance command must inherit both:

```
SPARK_DS41F_SUITE_MODEL_CONTRACT=/absolute/path/to/frozen-model-contract.json
SPARK_DS41F_SUITE_MODEL_CONTRACT_SHA256=<exact frozen file SHA256>
```

The contract is loaded and hash-checked by `command_suite`. Only its named suite
is opted in; companion suites continue normally. No opt-in environment leaves
the existing continue-after-error behavior unchanged. A source-bound candidate
image/commit/profile/rank set is mandatory, alongside both saved reference boots.

Before the existing model operation, the barrier verifies both saved C40 file
paths, exact encoded-byte hashes/sizes, source declarations and model shape. It
derives compact generation and teacher-forced score fingerprints from those
saved files inside the future owned window. It never runs a reference or adds
a candidate request. Both saved boots must give identical exact fingerprints;
neither can be selected away. Missing or inconsistent reference/identity input
fails before the candidate model starts and keeps a preflight error receipt.

The saved source is C40 head5f926df6/treeca1496c6, image7ae96e98, execution1f134c7a.
Root supplied the immutable native paths, byte sizes and hashes from its published
export receipt. The template additionally freezes
`runs/preflight/model-reference-provenance.json` by exact hash. That small record
binds both artifacts to the published analysis/export hashes and all three rank
start timestamps per boot; reusing any rank's boot is refused. Their expected corpus is
`5f7d25fcf89f62ce2877a941d3eabf9ca0c8f77587e35a7b989e24cf02385a60`,
with256 sequences and65,536 scored positions. Only those small metadata records
were read on the workstation; the15MB saved model files were not accessed.

Immediately after the existing model step, the barrier retains its native exit
and output and writes `model.json.gz.evidence.json`. It requires complete unique
prompt IDs, generated/scored token agreement, complete finite score arrays,
valid corpus identity, exact workload and generation/score equality with both
saved C40 boots. The compact score digest includes the existing comparison's
given-token probabilities and ordered top-token distributions, plus token IDs.
Rank shape, validity and digest are observations and do not enter equality.
Finite numbers use exact integer-ratio normalization, so signed zero and
equivalent integer/float values match the existing comparator without a new
epsilon. Self-agreement metadata is not compared: the repeat normally scores
the first boot's corpus and records null. Raw output and native exits are
preserved; an incomplete parse truthfully records that its full hash was not completed.

Either a nonzero/exceptional model exit or zero-exit semantic failure stops before
grid, records `stopped_before_timing`, and returns a failed suite result. The lab
must also stop companion suites/repeat boots on that suite failure; root is
connecting C46's separate `fail_on_suite_error` policy. The built-in release hash
`4f54f4b56f8d323473c7725d428caa2a4a76e04b9e728cb374fd092115669f03`,
its model/grid order, and all26 planned two-boot operations stay unchanged.

This is an exact cleanup-quality contract, not a serving-speed claim. The PR934
functional evidence API independently handles the three greedy requirements and
sampled isolated repeat while retaining C8 batch differences as observations.
Root's private C46 post-functional audit still covers the remaining C40 contract
and the final six-suite/26-step/conditions/identity audit remains required.

## Parser and failure corrections awaiting remote checks

The earlier nine-case remote CPU proof is retained. These subsequent corrections
have not been executed. Production parsing now runs only in fresh Linux parser
subprocesses, leaving the benchmark client's affinity and limits unchanged. Each
parser uses one allowed CPU, at most4GiB virtual address space (or a stricter
inherited limit),30/35 CPU seconds and60 wall seconds. Encoded files are limited
to32MiB before decoding and while reading; decoded reads stop at256MiB. Output
receipts are bounded too. These are parser-process VMS/CPU/read bounds, not an
RSS, cgroup or no-swap claim; the existing host memory/swap guards remain required.

The native model step outcome is written before evaluation. EOF, memory,
recursion, timeout or signal failures become failed evidence and stop before
grid; the parent performs no unbounded fallback scan. A model-containing suite
with a mismatched contract fails before requests. Model-free companion suites
remain allowed, and suites without the opt-in keep their prior behavior.
