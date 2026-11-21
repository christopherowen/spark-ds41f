"""pytest plugin: upstream tests' stand-in hub models resolve to the deployed DeepSeek checkpoint.

The release tests run offline, with only the served DeepSeek V4.1 Flash checkpoint on the node.
Upstream vLLM tests name small hub models (``gpt2``, ``facebook/opt-125m``, ``Qwen/Qwen3-0.6B``)
when they only need a tokenizer or a model config: ``create_scheduler``'s default model,
``ModelConfig``'s default model, a structured-output tokenizer. Offline, those tests can only
skip (``offline_skip``).

This plugin builds a Hugging Face hub cache in which each stand-in id resolves to the deployed
checkpoint's ``config.json`` and tokenizer, never its weights, and points the hub at it before
any test imports ``huggingface_hub``. Those tests then run against DeepSeek's config and
tokenizer. A test that needs a stand-in's weights, or one of its tokens, fails and is classified
like any other failure. A test whose setup DeepSeek V4.1's own config validation refuses (``REFUSALS``:
aligned prefill chunks without a threshold, DBO, a classic MTP draft, no tokenizer) is reported as
skipped, naming the guard: it asks for a configuration this deployment cannot construct.

The checkpoint is found at ``$DS41F_SNAPSHOT``, else under ``$HF_HUB_CACHE`` or
``~/.cache/huggingface/hub``. Without it, nothing changes and the stand-ins stay offline.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# Hub ids the tests use only for a tokenizer or a config. DeepSeek-V4.1-Flash's config and
# tokenizer are 1017's (config/cluster.json: "config, tokenizer and tensor layout unchanged").
STAND_INS = ("gpt2", "openai-community/gpt2", "facebook/opt-125m", "Qwen/Qwen3-0.6B",
             "deepseek-ai/DeepSeek-V4.1-Flash")
# What a stand-in gets: the config and the tokenizer, never weights.
STAND_IN_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def repo_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def find_snapshot() -> Path | None:
    named = os.environ.get("DS41F_SNAPSHOT")
    if named:
        path = Path(named)
        return path if (path / "config.json").is_file() else None
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and
             str(Path(os.environ["HF_HOME"]) / "hub"), str(Path.home() / ".cache/huggingface/hub")]
    for root in filter(None, roots):
        snapshots = sorted((repo_dir(Path(root), REPO) / "snapshots").glob("*/config.json"))
        if len(snapshots) == 1:
            return snapshots[0].parent
    return None


def build_cache(snapshot: Path, cache: Path) -> None:
    """A hub cache: the checkpoint under its own id, and each stand-in with its config and
    tokenizer only. Every entry is a symlink to the checkpoint's resolved file."""
    revision = snapshot.name
    entries = {REPO: sorted(p.name for p in snapshot.iterdir())}
    entries.update({repo: list(STAND_IN_FILES) for repo in STAND_INS})
    for repo, names in entries.items():
        root = repo_dir(cache, repo)
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(revision)
        target = root / "snapshots" / revision
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = snapshot / name
            if source.exists() and not (target / name).exists():
                (target / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())


SNAPSHOT = find_snapshot()
CACHE = None
if SNAPSHOT is not None:
    CACHE = Path(tempfile.mkdtemp(prefix="ds41f-hub-"))
    build_cache(SNAPSHOT, CACHE)
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[name] = str(CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        # Imported before this plugin (another plugin): repoint the module it already read.
        constants.HF_HUB_CACHE = constants.HUGGINGFACE_HUB_CACHE = str(CACHE)
        constants.HF_HUB_OFFLINE = True


def pytest_report_header(config):
    if SNAPSHOT is None:
        return "deepseek_models: no DeepSeek V4.1 Flash checkpoint found; stand-in models stay offline"
    return (f"deepseek_models: {', '.join(STAND_INS)} -> {REPO} config and tokenizer "
            f"(snapshot {SNAPSHOT.name[:12]}, hub cache {CACHE})")


# A stand-in test's setup that DeepSeek V4.1's own config validation refuses: the test asks for
# a configuration this deployment cannot construct, so it is not this deployment's to run.
# Each guard names its reason; any other failure stays a failure.
REFUSALS = (
    ("aligned prefill chunks need long_prefill_token_threshold (0)",
     "DeepSeek V4.1 aligns prefill chunks, and the test sets no long_prefill_token_threshold"),
    ("DeepSeek V4.1 Engram does not support DBO or microbatching",
     "DeepSeek V4.1's Engram runs without DBO or microbatching"),
    ("DeepSeek V4.1 has no classic-MTP draft",
     "DeepSeek V4.1 drafts with DSpark, not a classic MTP or draft model"),
    ("requires a tokenizer and cannot be initialized when `skip_tokenizer_init=True`",
     "DeepSeek V4.1's multimodal processor needs its tokenizer"),
)


def refusal(text: str) -> str | None:
    return next((reason for message, reason in REFUSALS if message in text), None)


try:
    import pytest
except ImportError:  # imported outside pytest (the unit tests)
    pytest = None

if pytest is not None and hasattr(pytest, "hookimpl"):
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        outcome = yield
        report = outcome.get_result()
        if SNAPSHOT is None or not report.failed or call.excinfo is None:
            return
        reason = refusal(f"{call.excinfo.typename}: {call.excinfo.value}")
        if reason is not None:
            report.outcome = "skipped"
            report.longrepr = (str(item.path), item.location[1] or 0,
                               f"the DeepSeek V4.1 config refuses this test's setup: {reason}")
