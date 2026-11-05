# Commands and receipts

Deployment checkout on Spaceport:
`/home/chris/spark-lab/checkouts/drafter-orchestration-20261105`.
Source checkout:
`/home/chris/spark-lab/checkouts/vllm-drafter-orchestration-20261105`.
Remote work uses `flock /home/chris/spark-cpu-work.lock`, `taskset -c 0,1`,
`nice -n 10`, an 8-GiB virtual-memory bound for source preparation/lint, and
`MAX_JOBS=2 OMP_NUM_THREADS=2`. CPU containers have `--cpus=2 --memory=8g`,
`NVIDIA_VISIBLE_DEVICES=void` and empty `CUDA_VISIBLE_DEVICES`.

## Source preparation

In the deployment checkout:

```sh
python3 scripts/example_site.py
bin/spark doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/tp3.json build prepare --only vllm
bin/spark --cluster-config experiments/2026-11-05-drafter-metadata-replay/candidate.json doctor
bin/spark --cluster-config experiments/2026-11-05-drafter-metadata-replay/candidate.json build prepare --only vllm
git diff --check
```

Only disposable example site maps are used by these offline checks. Neither
source preparation command is a cluster sync or image build. Logs retain the
inherited intermediate patch whitespace warning; final trees are checked.
The earlier workstation preparation failure is recorded in `runs/setup-events.json`.

## Host tests

The CPU test container used the immutable candidate-38 image in `base.json`,
with the named source checkout mounted at `/source`. Its isolated environment:

```sh
uv venv --python /usr/bin/python3 --system-site-packages /tmp/drafter-venv
uv pip install --python /tmp/drafter-venv/bin/python -r /source/requirements/lint.txt tblib pytest-timeout ruff==0.14.0
cd /source
/tmp/drafter-venv/bin/python -c 'import vllm.platforms; from vllm.platforms.cpu import CpuPlatform; vllm.platforms.current_platform=CpuPlatform(); import vllm.tilelang_utils as tl; tl._ensure_tilelang_imported(); import pytest; raise SystemExit(pytest.main(["--noconftest", "-p", "no:cacheprovider", "-q", "tests/v1/cudagraph/test_cudagraph_manager.py"]))'
/tmp/drafter-venv/bin/python -c 'import vllm.platforms; from vllm.platforms.cpu import CpuPlatform; vllm.platforms.current_platform=CpuPlatform(); import vllm.tilelang_utils as tl; tl._ensure_tilelang_imported(); import pytest; raise SystemExit(pytest.main(["--noconftest", "-p", "no:cacheprovider", "-q", "tests/v1/cudagraph/test_cudagraph_manager.py", "tests/v1/attention/test_b12x_v41_ced_metadata.py", "-k", "dflash or draft_metadata_graph"]))'
```

This explicitly selects the CPU platform and imports TileLang definitions without
launching kernels. `--noconftest` omits unrelated model fixtures, not the fixtures
in the tested graph file. The full graph suite passed 35 tests; final focused
coverage passed 9, skipped the 3 CUDA cases and deselected 65 unrelated tests.
The default unspecified-platform attempt had 33 passes and 2 host stream failures;
it remains in `cpu-tests-2.log`. These are not GPU correctness results.

## Source checks

On the host, a separate `uv` environment installs `requirements/lint.txt` and
Ruff 0.14.0. Git and patch objects stay on Spaceport. Run in the source checkout:

```sh
.venv/bin/pre-commit install
RUFF_CACHE_DIR=/tmp/drafter-ruff-host .venv/bin/pre-commit run --files vllm/models/deepseek_v4_1/sparse_mla.py vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py vllm/v1/worker/gpu/spec_decode/dflash/speculator.py tests/v1/cudagraph/test_cudagraph_manager.py tests/v1/attention/test_b12x_v41_ced_metadata.py
git diff --check
```

`source-lint-1..4.log` retain every attempt. Two formatting adjustments and an
existing memory-query call in touched files were normalized to satisfy the
repository hooks. All applicable hooks pass, including Ruff, mypy and accelerator
API checks. The source commit reran those hooks successfully (`source-commit.log`).

## Deployment repository tests

From the isolated deployment checkout on Spaceport:

```sh
/home/chris/spark-lab/checkouts/vllm-drafter-orchestration-20261105/.venv/bin/python -m unittest discover -s tests
```

Result: 755 tests, 5 skipped, OK in 44.770 seconds. The native log includes
synthetic benchmark fixtures against test servers. None are serving or GPU
measurements, and none enter this experiment's performance results.
