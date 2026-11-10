"""bin/spark candidate: the candidate process's orchestration steps, from the owner's laptop.

Only the laptop reaches both the dgx and the spaceport networks, so these commands run there
and drive each testbed's head over ssh (docs/lab.md, "Candidate tooling";
docs/release-benchmark.md, "Candidates"). Every remote command runs under ``bash -lc`` on the
node, built with ``shlex.quote``; every ssh call has a connect timeout, an overall timeout and
retries, and a remote read ends with a sentinel, so a dropped connection is retried or raised and
never reads as an empty result. Commands that change anything print their plan and act only
with ``--apply``.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import recipe  # noqa: E402  (scripts/recipe.py: the pure overlay and shape resolution)

CANDIDATES = "experiments/2026-10-26-spark-ds41f-candidate"
PRODUCTION_CONFIG = "config/cluster.json"
RESULTS = "results/private"
LAB = f"{RESULTS}/lab"
QUEUE = f"{LAB}/queue"
QUEUE_HOLDER = "spark-lab: queue"  # the hold a lab queue runner writes (scripts/lab.py queue run)
SOURCE_ROOT = "/opt/spark-ds41f/candidate"  # the image's source trees; vLLM's is <root>/vllm
SHAPE_NAMES = {"tp4": "tp4-ring", "tp3": "tp3-triangle"}  # profile stem -> recipe shape
SSH_OPTIONS = ("-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
END = "@@candidate-end"


@dataclasses.dataclass(frozen=True)
class Testbed:
    name: str
    head: str
    nodes: tuple[str, ...]
    profile: str  # the shape profile in a candidate folder
    checkout: str = "~/projects/spark-ds41f"

    @property
    def shape(self) -> str:
        return self.profile.removesuffix(".json")


# The testbeds (docs/release-benchmark.md, "Testbeds"). The head is the first node.
TESTBEDS = {testbed.name: testbed for testbed in (
    Testbed("tp4-ring", "dgx1", ("dgx1", "dgx2", "dgx3", "dgx4"), "tp4.json"),
    Testbed("tp4-switched", "sp1", ("sp1", "sp2", "sp3", "sp4"), "tp4-switched.json"),
    Testbed("tp3-switched", "sp5", ("sp5", "sp6", "sp7"), "tp3-switched.json"),
)}
ONE_GPU_NODE = "sp8"  # each PR's own benchmark; not a queue testbed


class RemoteError(RuntimeError):
    """An ssh call that failed after its retries: never an empty result."""


class Refused(RuntimeError):
    """A guard refused the operation; nothing (further) was changed."""


def say(message: str = "") -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- ssh

def home_path(path: str) -> str:
    """A node path for the remote shell: ``~/x`` becomes ``"$HOME"/x`` (quoted), others are quoted."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def in_checkout(testbed: Testbed, command: str) -> str:
    return f"cd {home_path(testbed.checkout)} && {command}"


@dataclasses.dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


class Ssh:
    """ssh with argument lists only: the remote command is one ``bash -lc <quoted script>`` string."""

    def __init__(self, run: Callable = subprocess.run, popen: Callable = subprocess.Popen,
                 attempts: int = 4, delay: float = 10.0, timeout: float = 120.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.run, self.popen = run, popen
        self.attempts, self.delay, self.timeout, self.sleep = attempts, delay, timeout, sleep

    @staticmethod
    def argv(host: str, script: str) -> list[str]:
        return ["ssh", *SSH_OPTIONS, host, "bash -lc " + shlex.quote(script)]

    @staticmethod
    def wrap(script: str) -> str:
        # A subshell, so an ``exit`` in the script still reaches the sentinel with its status.
        return f"(\n{script}\n)\n__rc=$?\nprintf '\\n{END} %s\\n' \"$__rc\""

    @staticmethod
    def unwrap(stdout: str) -> tuple[int, str] | None:
        marker = f"\n{END} "
        index = stdout.rfind(marker)
        if index < 0:
            return None
        tail = stdout[index + len(marker):].split()
        if not tail or not tail[0].lstrip("-").isdigit():
            return None
        return int(tail[0]), stdout[:index]

    def call(self, host: str, script: str, *, timeout: float | None = None, retry: bool = True) -> Result:
        """Run ``script`` on ``host``. An ssh failure (exit 255), a timeout or output cut short is
        retried when ``retry`` (reads and idempotent steps), else raised: never an empty result."""
        attempts = self.attempts if retry else 1
        problem = "no attempt"
        for attempt in range(1, attempts + 1):
            try:
                process = self.run(self.argv(host, self.wrap(script)), capture_output=True, text=True,
                                   timeout=timeout or self.timeout, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                problem = f"timed out after {timeout or self.timeout:g} s"
            else:
                unwrapped = self.unwrap(process.stdout or "")
                if process.returncode == 255:
                    problem = f"ssh failed: {(process.stderr or '').strip()[-300:] or 'exit 255'}"
                elif unwrapped is None:
                    problem = f"output cut short (exit {process.returncode})"
                else:
                    return Result(unwrapped[0], unwrapped[1], process.stderr or "")
            if attempt < attempts:
                warn(f"{host}: {problem}; retrying ({attempt}/{attempts - 1})")
                self.sleep(self.delay)
        raise RemoteError(f"{host}: {problem}" + (f" after {attempts} attempts" if attempts > 1 else ""))

    def stream(self, host: str, script: str, consume: Callable, *, timeout: float) -> None:
        """Run ``script`` on ``host`` and hand its binary stdout to ``consume``; ``consume`` must
        start afresh on every attempt (the stream is retried as a whole)."""
        problem = "no attempt"
        for attempt in range(1, self.attempts + 1):
            process = self.popen(self.argv(host, script), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 stdin=subprocess.DEVNULL)
            timer = threading.Timer(timeout, process.kill)
            timer.start()
            try:
                consume(process.stdout)
                process.stdout.read()
                error = process.stderr.read().decode(errors="replace")
                code = process.wait()
            except (OSError, tarfile.TarError, ValueError) as caught:
                process.kill()
                process.wait()
                code, error = -1, str(caught)
            finally:
                timer.cancel()
            if code == 0:
                return
            problem = f"stream exited {code}: {error.strip()[-300:]}"
            if attempt < self.attempts:
                warn(f"{host}: {problem}; retrying ({attempt}/{self.attempts - 1})")
                self.sleep(self.delay)
        raise RemoteError(f"{host}: {problem} after {self.attempts} attempts")

    def pipe(self, source: str, source_script: str, target: str, target_script: str,
             *, timeout: float) -> tuple[bool, str]:
        """``ssh source source_script | ssh target target_script`` through this host, once."""
        producer = self.popen(self.argv(source, source_script), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        consumer = self.popen(self.argv(target, target_script), stdin=producer.stdout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        producer.stdout.close()
        timer = threading.Timer(timeout, lambda: (producer.kill(), consumer.kill()))
        timer.start()
        try:
            out, err = consumer.communicate()
            produced_err = producer.stderr.read()
            producer.wait()
        finally:
            timer.cancel()
        ok = producer.returncode == 0 and consumer.returncode == 0
        message = (out or b"").decode(errors="replace") + (err or b"").decode(errors="replace") \
            + (produced_err or b"").decode(errors="replace")
        return ok, message.strip()


def sections(text: str, marker: str = "@@section ") -> dict[str, list[str]]:
    """Output lines grouped under ``@@section NAME`` lines; lines before the first are dropped
    (a login shell's own output)."""
    found: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        if line.startswith(marker):
            name = line[len(marker):].strip()
            found[name] = []
        elif name is not None:
            found[name].append(line)
    return found


# ---------------------------------------------------------------- local repository

class Local:
    """The laptop's checkout: fetch, and read files at a revision (what a node detaches at)."""

    def __init__(self, root: Path | None = None):
        self.root = root or ROOT

    def git(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.root), *arguments], text=True, capture_output=True,
                              check=check, timeout=300)

    def fetch(self) -> None:
        fetched = self.git("fetch", "-q", "origin", check=False)
        if fetched.returncode:
            raise Refused(f"git fetch origin failed in {self.root}: {fetched.stderr.strip()[-300:]}")

    def rev_parse(self, ref: str) -> str | None:
        found = self.git("rev-parse", "--verify", "-q", f"{ref}^{{commit}}", check=False)
        return found.stdout.strip() if found.returncode == 0 else None

    def reader(self, revision: str | None) -> Callable[[str], str | None]:
        """Read a repository-relative file at ``revision``, or from the working tree with None."""
        def read(path: str) -> str | None:
            if revision is None:
                file = self.root / path
                return file.read_text(encoding="utf-8") if file.is_file() else None
            shown = self.git("show", f"{revision}:{path}", check=False)
            return shown.stdout if shown.returncode == 0 else None
        return read


# ---------------------------------------------------------------- lint

HEX = re.compile(r"(?<![0-9A-Za-z_])[0-9a-fA-F]{7,40}(?![0-9A-Za-z_])")


def cited_hashes(text: str) -> list[str]:
    """Hex words of 7-40 characters with a digit and a letter: commit and tree hashes, not
    numbers (1048576) or words (defaced)."""
    return [word for word in HEX.findall(text)
            if re.search(r"[0-9]", word) and re.search(r"[a-fA-F]", word)]


def newline_problem(path: str, text: str) -> str | None:
    if not text.endswith("\n"):
        return f"{path}: does not end in a newline"
    if text.endswith("\n\n"):
        return f"{path}: ends in more than one newline"
    return None


def arm_path(experiment: str, config: str) -> str:
    """As the lab resolves an arm: a bare name is the experiment's, a path is repository-relative."""
    return config if "/" in config else f"{experiment}/{config}"


def resolve_profile(read: Callable[[str], str | None], path: str, shape: str | None = None,
                    depth: int = 0) -> dict:
    """A profile as concrete settings, read through ``read`` (a revision or the working tree)."""
    if depth > 16:
        raise ValueError(f"{path}: extends chain too deep")
    text = read(path)
    if text is None:
        raise ValueError(f"{path} does not exist")
    data = json.loads(text)
    try:
        if recipe.is_overlay(data):
            base = data["extends"]
            return recipe.overlay(data, resolve_profile(read, base["config"], base.get("shape"), depth + 1))
        if recipe.is_recipe(data):
            if not shape:
                raise ValueError(f"{path} is a recipe; the arm that extends it must name its shape")
            return recipe.resolve(data, shape)
    except SystemExit as error:  # recipe's refusals
        raise ValueError(f"{path}: {error}") from None
    return data


def tree_labels(cluster: dict, read: Callable[[str], str | None] | None = None) -> list[str]:
    """The image's own identity a note may cite: its tree labels, and the patch heads and
    trees of the profile's own upstream lock (which build checks against the image)."""
    labels = cluster.get("container", {}).get("expected_labels", {})
    own = [str(value).lower() for key, value in labels.items() if key.endswith(".tree")]
    lock = cluster.get("upstreams_config")
    text = read(lock) if read and isinstance(lock, str) else None
    if text:
        try:
            sources = json.loads(text).get("sources", {})
        except ValueError:
            sources = {}
        for source in sources.values():
            if isinstance(source, dict):
                own += [str(source[key]).lower() for key in ("patch_head", "expected_tree") if source.get(key)]
    return own


def notes_problems(path: str, notes: object, allowed: Iterable[str] = ()) -> list[str]:
    if not isinstance(notes, str):
        return []
    allowed = [label.lower() for label in allowed]
    return [f"{path}: notes cite {word}, which is not the image's own tree label (or its lock's patch head)"
            for word in cited_hashes(notes) if not any(label.startswith(word.lower()) for label in allowed)]


def lint_profile(read: Callable[[str], str | None], path: str, seen: set | None = None) -> list[str]:
    """A profile: it exists, parses, ends in one newline, and its notes cite no hash but the
    image's own tree labels; the same for every profile it extends under experiments/."""
    seen = set() if seen is None else seen
    if path in seen:
        return []
    seen.add(path)
    text = read(path)
    if text is None:
        return [f"{path}: profile does not exist"]
    problems = [problem for problem in [newline_problem(path, text)] if problem]
    try:
        data = json.loads(text)
    except ValueError as error:
        return problems + [f"{path}: not JSON ({error})"]
    try:
        labels = tree_labels(resolve_profile(read, path), read)
    except (ValueError, KeyError, TypeError) as error:
        problems.append(f"{path}: cannot resolve ({error})")
        labels = []
    problems += notes_problems(path, data.get("notes"), labels)
    base = data.get("extends", {}).get("config") if isinstance(data.get("extends"), dict) else None
    if base and base.startswith("experiments/"):
        problems += lint_profile(read, base, seen)
    return problems


def spec_profiles(spec: dict) -> list[str]:
    """Every profile a lab spec boots or builds: arms, an agentx arm, and a job's own config."""
    experiment = spec.get("experiment", "")
    paths = []
    for job in spec.get("jobs", []):
        for arm in job.get("arms", []) + ([job["arm"]] if isinstance(job.get("arm"), dict) else []):
            paths.append(arm_path(experiment, arm["config"]))
        if isinstance(job.get("config"), str):
            paths.append(arm_path(experiment, job["config"]))
    return list(dict.fromkeys(paths))


def lint_spec(read: Callable[[str], str | None], path: str, text: str | None = None) -> list[str]:
    text = read(path) if text is None else text
    if text is None:
        return [f"{path}: spec does not exist"]
    problems = [problem for problem in [newline_problem(path, text)] if problem]
    try:
        spec = json.loads(text)
    except ValueError as error:
        return problems + [f"{path}: not JSON ({error})"]
    if not isinstance(spec.get("run"), str) or not isinstance(spec.get("jobs"), list):
        problems.append(f"{path}: a spec names a run and its jobs")
        return problems
    seen: set = set()
    for profile in spec_profiles(spec):
        if read(profile) is None:
            problems.append(f"{path}: references {profile}, which does not exist")
        else:
            problems += lint_profile(read, profile, seen)
    return problems


def lint_paths(paths: list[str], read: Callable[[str], str | None]) -> list[str]:
    problems = []
    for path in paths:
        text = read(path)
        if text is None:
            problems.append(f"{path}: does not exist")
            continue
        if path.endswith(".md"):
            problems += [problem for problem in [newline_problem(path, text)] if problem]
        elif path.endswith(".json"):
            try:
                data = json.loads(text)
            except ValueError as error:
                problems.append(f"{path}: not JSON ({error})")
                continue
            if isinstance(data, dict) and "jobs" in data:
                problems += lint_spec(read, path, text)
            elif isinstance(data, dict) and ("extends" in data or "container" in data or "notes" in data):
                problems += lint_profile(read, path)
            else:
                problems += [problem for problem in [newline_problem(path, text)] if problem]
    return list(dict.fromkeys(problems))


def checked_json(path: str, data: dict) -> str:
    """The text the tool writes for a JSON file: one trailing newline, notes with no hash."""
    text = json.dumps(data, indent=1) + "\n"
    problems = [problem for problem in [newline_problem(path, text)] if problem]
    problems += notes_problems(path, data.get("notes"))
    if problems:
        raise Refused("; ".join(problems))
    return text


def repo_relative(path: str) -> str:
    """A repository-relative path from a path given relative to the repository or absolute in it."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            raise Refused(f"{path} is outside the repository") from None
    return str(Path(os.path.normpath(path)))


# ---------------------------------------------------------------- queue state

@dataclasses.dataclass
class QueueState:
    testbed: Testbed
    hold: dict | None
    head: str
    pending: list[str]
    running: list[str]
    done: list[tuple[int, str]]  # (ctime, name): ctime is when the runner moved it there
    failed: list[tuple[int, str]]
    watchers: list[str]
    runners: list[str]
    containers: list[str]
    runs: dict[str, str]  # pending or running entry -> its spec's run

    @property
    def hold_is_queue(self) -> bool:
        return bool(self.hold) and str(self.hold.get("holder", "")).startswith(QUEUE_HOLDER)

    @property
    def foreign_hold(self) -> bool:
        return bool(self.hold) and not self.hold_is_queue

    @property
    def busy(self) -> bool:
        """A job runs from the checkout, or a queue runner in an open window is about to start one."""
        return bool(self.running) or (bool(self.pending) and self.hold_is_queue)


def state_script() -> str:
    lines = [
        "echo @@section hold",
        'cat "$HOME"/spark-hold.json 2>/dev/null || cat "$HOME"/spark3-hold.json 2>/dev/null',
        "echo @@section head",
        "git rev-parse HEAD",
    ]
    for state, directory in (("pending", QUEUE), ("running", f"{QUEUE}/running"),
                             ("done", f"{QUEUE}/done"), ("failed", f"{QUEUE}/failed")):
        lines += [f"echo @@section {state}",
                  f"( cd {shlex.quote(directory)} 2>/dev/null && for f in *.json; do "
                  "[ -f \"$f\" ] && stat -c '%Z %n' -- \"$f\"; done; true )"]
    lines += [
        "echo @@section entries",
        f"for f in {shlex.quote(QUEUE)}/*.json {shlex.quote(QUEUE)}/running/*.json; do "
        '[ -f "$f" ] || continue; echo "@@entry ${f##*/}"; grep -m1 \'"run":\' -- "$f"; done',
        "echo @@section watchers",
        # The bracket keeps pgrep from matching this command's own bash.
        "pgrep -af '[l]ab_idle_start' || true",
        "echo @@section runners",
        "pgrep -af '[l]ab\\.py queue run' || true",
        "echo @@section containers",
        "docker ps --format " + shlex.quote('{{.Names}}|{{.Image}}|{{.Status}}|{{.Label "spark.cluster-config"}}')
        + " 2>&1 || true",
    ]
    return "\n".join(lines)


def parse_state(testbed: Testbed, text: str) -> QueueState:
    found = sections(text)
    missing = [name for name in ("hold", "head", "pending", "running", "done", "failed", "containers")
               if name not in found]
    if missing:
        raise RemoteError(f"{testbed.head}: state output lacks {', '.join(missing)}")
    hold_text = "\n".join(found["hold"]).strip()
    hold = None
    if hold_text:
        try:
            hold = json.loads(hold_text)
        except ValueError:
            hold = {"holder": "unreadable hold file"}

    def listed(name: str) -> list[tuple[int, str]]:
        entries = []
        for line in found[name]:
            stamp, _, entry = line.partition(" ")
            if entry.endswith(".json") and stamp.isdigit():
                entries.append((int(stamp), entry))
        return entries

    runs, current = {}, None
    for line in found.get("entries", []):
        if line.startswith("@@entry "):
            current = line[len("@@entry "):].strip()
        elif current and '"run":' in line:
            match = re.search(r'"run":\s*"((?:[^"\\]|\\.)*)"', line)
            if match:
                runs[current] = json.loads(f'"{match.group(1)}"')
    return QueueState(
        testbed=testbed, hold=hold, head=(found["head"] or [""])[0].strip(),
        pending=sorted(entry for _, entry in listed("pending")),
        running=sorted(entry for _, entry in listed("running")),
        done=sorted(listed("done")), failed=sorted(listed("failed")),
        watchers=[line for line in found.get("watchers", []) if line.strip()],
        runners=[line for line in found.get("runners", []) if line.strip()],
        containers=[line for line in found["containers"] if line.strip()], runs=runs)


def read_state(ssh: Ssh, testbed: Testbed) -> QueueState:
    result = ssh.call(testbed.head, f"cd {home_path(testbed.checkout)} || exit 3\n{state_script()}")
    if result.returncode == 3:
        raise RemoteError(f"{testbed.head}: no checkout at {testbed.checkout}")
    return parse_state(testbed, result.stdout)


def heartbeat_age(hold: dict) -> str:
    try:
        stamp = time.mktime(time.strptime(hold["heartbeat"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        return f"{time.time() - stamp:.0f} s ago"
    except (KeyError, ValueError, TypeError):
        return "unknown"


def describe_state(state: QueueState) -> list[str]:
    testbed = state.testbed
    lines = [f"{testbed.name} ({testbed.head}, checkout {state.head[:12] or '?'})"]
    if state.hold:
        lines.append(f"  hold:         {state.hold.get('holder')} (heartbeat {heartbeat_age(state.hold)})")
        lines.append(f"  current:      {state.hold.get('current', '-')}")
    else:
        lines.append("  hold:         none")
    lines.append(f"  running:      {', '.join(state.running) or '-'}")
    lines.append(f"  pending:      {', '.join(state.pending) or '-'}")
    for name, entries in (("last done:   ", state.done), ("last failed: ", state.failed)):
        if entries:
            stamp, entry = max(entries)
            lines.append(f"  {name} {entry} ({time.strftime('%Y-%m-%d %H:%MZ', time.gmtime(stamp))})")
        else:
            lines.append(f"  {name} -")
    watcher = next((line for line in state.watchers if "bash -c" not in line), None) or \
        (state.watchers[0] if state.watchers else None)
    lines.append(f"  idle watcher: {'running: ' + watcher if watcher else 'none'}")
    runner = next((line for line in state.runners if "bash -c" not in line), None) or \
        (state.runners[0] if state.runners else None)
    lines.append(f"  queue runner: {'running: ' + runner if runner else 'none'}")
    if not state.containers:
        lines.append("  container:    none running")
    for line in state.containers:
        name, image, status, profile = (line.split("|") + ["", "", ""])[:4]
        lines.append(f"  container:    {name} {image} ({status})" + (f" from {profile}" if profile else ""))
    return lines


def testbeds_of(names: list[str] | None) -> list[Testbed]:
    unknown = [name for name in names or [] if name not in TESTBEDS]
    if unknown:
        raise Refused(f"unknown testbed {', '.join(unknown)}; testbeds: {', '.join(TESTBEDS)}")
    return [TESTBEDS[name] for name in names] if names else list(TESTBEDS.values())


def command_status(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    code = 0
    for testbed in testbeds_of(args.testbeds):
        try:
            say("\n".join(describe_state(read_state(ssh, testbed))))
        except RemoteError as error:
            say(f"{testbed.name} ({testbed.head}): unreachable: {error}")
            code = 1
    return code


# ---------------------------------------------------------------- wait

def finished_script() -> str:
    lines = []
    for state, directory in (("done", f"{QUEUE}/done"), ("failed", f"{QUEUE}/failed")):
        lines += [f"echo @@section {state}",
                  f"( cd {shlex.quote(directory)} 2>/dev/null && for f in *.json; do "
                  "[ -f \"$f\" ] && echo \"$f\"; done; true )"]
    return "\n".join(lines)


def finished_sets(ssh: Ssh, testbed: Testbed) -> tuple[frozenset, frozenset]:
    """The done and failed entry names, each from its own directory (never running/)."""
    result = ssh.call(testbed.head, f"cd {home_path(testbed.checkout)} || exit 3\n{finished_script()}")
    found = sections(result.stdout)
    if result.returncode or "done" not in found or "failed" not in found:
        raise RemoteError(f"{testbed.head}: cannot list the queue (exit {result.returncode})")
    pick = lambda name: frozenset(line.strip() for line in found[name] if line.strip().endswith(".json"))
    return pick("done"), pick("failed")


def finished_changes(before: dict, after: dict) -> dict[str, dict[str, list[str]]]:
    """Entries that arrived in done or failed since ``before``, per testbed."""
    changes = {}
    for name, (done, failed) in after.items():
        if name not in before:
            continue
        new_done, new_failed = sorted(done - before[name][0]), sorted(failed - before[name][1])
        if new_done or new_failed:
            changes[name] = {"done": new_done, "failed": new_failed}
    return changes


def command_wait(args: argparse.Namespace, ssh: Ssh | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    ssh = ssh or Ssh()
    testbeds = testbeds_of(args.testbeds)
    deadline = clock() + args.timeout if args.timeout > 0 else None
    baseline: dict = {}
    while True:
        for testbed in testbeds:
            if testbed.name in baseline:
                continue
            try:
                baseline[testbed.name] = finished_sets(ssh, testbed)
            except RemoteError as error:
                warn(f"{testbed.name}: {error}; will retry")
        if len(baseline) == len(testbeds):
            break
        if deadline is not None and clock() >= deadline:
            say("timed out before every queue could be read")
            return 2
        sleep(args.interval)
    say(f"waiting for a job to finish on {', '.join(t.name for t in testbeds)}")
    while True:
        if deadline is not None and clock() >= deadline:
            say("timed out: nothing finished")
            return 2
        sleep(args.interval)
        current = {}
        for testbed in testbeds:
            try:
                current[testbed.name] = finished_sets(ssh, testbed)
            except RemoteError as error:
                warn(f"{testbed.name}: {error}; polling again")  # a failed read is no change
        changes = finished_changes(baseline, current)
        if not changes:
            continue
        for name, change in changes.items():
            for entry in change["done"]:
                say(f"{name}: done   {entry}")
            for entry in change["failed"]:
                say(f"{name}: FAILED {entry}")
        for testbed in testbeds:
            try:
                say("\n".join(describe_state(read_state(ssh, testbed))))
            except RemoteError as error:
                say(f"{testbed.name}: state unreadable: {error}")
        return 0


# ---------------------------------------------------------------- queue

def fetch_detach_command() -> str:
    return "git fetch -q origin && git checkout -q --detach origin/main && git rev-parse HEAD"


def sync_command() -> str:
    return f"bin/spark --cluster-config {PRODUCTION_CONFIG} cluster sync --apply"


def dry_run_command(spec: str) -> str:
    return f"python3 scripts/lab.py run --dry-run {shlex.quote(spec)}"


def queue_add_command(spec: str) -> str:
    return f"python3 scripts/lab.py queue add {shlex.quote(spec)}"


def watcher_command(minutes: int, log_name: str) -> str:
    inner = (f"scripts/lab_idle_start.sh {int(minutes)} {shlex.quote(log_name + '-window')} "
             f">> {shlex.quote(f'{LAB}/{log_name}-idle.log')} 2>&1")
    return f"setsid -f bash -c {shlex.quote(inner)} < /dev/null > /dev/null 2>&1"


def last_commit(text: str) -> str | None:
    found = re.findall(r"^[0-9a-f]{40}$", text, flags=re.MULTILINE)
    return found[-1] if found else None


def default_log_name(specs: list[str]) -> str:
    for spec in specs:
        match = re.search(r"(candidate-\d+[^/]*)/", spec)
        if match:
            return match.group(1)
    return "candidate-queue"


def spec_runs(read: Callable[[str], str | None], specs: list[str]) -> dict[str, str]:
    runs = {}
    for spec in specs:
        try:
            runs[spec] = json.loads(read(spec) or "{}").get("run", "")
        except ValueError:
            runs[spec] = ""
    return runs


def node_files_script(paths: list[str]) -> str:
    return "\n".join(f'if [ -f {shlex.quote(p)} ]; then echo "@@file {p}"; cat -- {shlex.quote(p)}; '
                     f'echo; echo "@@eof"; else echo "@@missing {p}"; fi' for p in paths)


def parse_node_files(text: str) -> dict[str, str | None]:
    """``@@file``/``@@eof`` blocks (content gets one newline added by the script, removed here)."""
    files: dict[str, str | None] = {}
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("@@missing "):
            files[line[len("@@missing "):]] = None
        elif line.startswith("@@file "):
            name, body = line[len("@@file "):], []
            index += 1
            while index < len(lines) and lines[index] != "@@eof":
                body.append(lines[index])
                index += 1
            # The script's echo ends the last line, so the joined lines are the file exactly.
            files[name] = "\n".join(body)
        index += 1
    return files


def lint_or_refuse(read: Callable[[str], str | None], specs: list[str], where: str) -> None:
    problems = []
    for spec in specs:
        problems += lint_spec(read, spec)
    if problems:
        raise Refused(f"lint at {where} refused:\n  " + "\n  ".join(dict.fromkeys(problems)))
    say(f"  lint at {where}: {len(specs)} spec(s) and their profiles clean")


def queue_specs(ssh: Ssh, local: Local, testbed: Testbed, specs: list[str], *, apply: bool,
                idle_minutes: int = 10, log_name: str | None = None) -> int:
    """Queue specs on a testbed without ever moving a checkout a job runs from."""
    specs = [repo_relative(spec) for spec in specs]
    log_name = log_name or default_log_name(specs)
    state = read_state(ssh, testbed)
    head = testbed.head
    if state.foreign_hold:
        raise Refused(f"{testbed.name}: held by {state.hold.get('holder')!r} since {state.hold.get('since')}; "
                      "not queueing")
    local.fetch()
    origin = local.rev_parse("origin/main")
    queued_runs = {run for entry, run in state.runs.items() if entry in state.pending + state.running}

    def add_all() -> None:
        runs = spec_runs(local.reader(origin), specs)
        for spec in specs:
            if runs.get(spec) and runs[spec] in queued_runs:
                say(f"  {spec}: run {runs[spec]} is already pending or running; not queued again")
                continue
            say(f"  {head}$ {in_checkout(testbed, queue_add_command(spec))}")
            if not apply:
                continue
            try:
                result = ssh.call(head, in_checkout(testbed, queue_add_command(spec)), retry=False)
            except RemoteError as error:
                # The add may have run before the connection dropped: look before adding again.
                warn(f"{head}: {error}; checking whether {spec} was queued")
                again = read_state(ssh, testbed)
                if runs.get(spec) and runs[spec] in again.runs.values():
                    say(f"  {spec}: queued (confirmed after the dropped connection)")
                    continue
                result = ssh.call(head, in_checkout(testbed, queue_add_command(spec)), retry=False)
            output = (result.stdout + result.stderr).strip()
            say("    " + output.replace("\n", "\n    ") if output else "    (no output)")
            if result.returncode:
                raise Refused(f"{testbed.name}: queue add {spec} failed (exit {result.returncode})")
            queued_runs.add(runs.get(spec, ""))

    if state.busy:
        what = ", ".join(state.running) or f"pending {', '.join(state.pending)} in the open window"
        say(f"{testbed.name}: a job is running or about to ({what}); the checkout stays where it is "
            f"({state.head[:12]}). Each job's sync moves it to origin/main when it starts.")
        nodes = parse_node_files(ssh.call(head, in_checkout(testbed, node_files_script(specs))).stdout)
        read_origin = local.reader(origin)
        problems = []
        for spec in specs:
            theirs, ours = nodes.get(spec), read_origin(spec)
            if theirs is None:
                problems.append(f"{spec} is not in {head}'s checkout")
            elif ours is not None and theirs != ours:
                problems.append(f"{spec} in {head}'s checkout differs from origin/main")
            elif ours is None:
                problems.append(f"{spec} is not on origin/main")
        if problems:
            raise Refused(f"{testbed.name}: " + "; ".join(problems)
                          + ". A spec is queued from the node's checkout, which must not move while a job "
                          "runs: retry when the queue is idle (candidate wait).")
        lint_or_refuse(read_origin, specs, f"origin/main {origin[:12] if origin else '?'}")
        add_all()
        if not apply:
            say("plan only; pass --apply to queue")
        return 0

    say(f"{testbed.name}: nothing is running; the checkout moves to origin/main first")
    say(f"  {head}$ {in_checkout(testbed, fetch_detach_command())}")
    say(f"  {head}$ {in_checkout(testbed, sync_command())}")
    for spec in specs:
        say(f"  {head}$ {in_checkout(testbed, dry_run_command(spec))}   (refuse on any problem)")
    if not apply:
        lint_or_refuse(local.reader(origin), specs, f"local origin/main {origin[:12] if origin else '?'}")
        add_all()
        if not (state.hold_is_queue or state.watchers):
            say(f"  {head}$ {in_checkout(testbed, watcher_command(idle_minutes, log_name))}")
        say("plan only; pass --apply to queue")
        return 0
    detached = ssh.call(head, in_checkout(testbed, fetch_detach_command()), timeout=300)
    commit = last_commit(detached.stdout)
    if detached.returncode or not commit:
        raise Refused(f"{head}: fetch and detach failed (exit {detached.returncode}): "
                      f"{(detached.stdout + detached.stderr).strip()[-400:]}")
    say(f"  {head} detached at {commit[:12]}")
    if not local.rev_parse(commit):
        local.fetch()
    if not local.rev_parse(commit):
        raise Refused(f"{head} detached at {commit[:12]}, which this checkout cannot read after a fetch")
    origin = commit
    lint_or_refuse(local.reader(commit), specs, f"{commit[:12]}")
    synced = ssh.call(head, in_checkout(testbed, sync_command()), timeout=1800)
    say("    " + (synced.stdout + synced.stderr).strip().replace("\n", "\n    "))
    if synced.returncode:
        raise Refused(f"{testbed.name}: cluster sync failed (exit {synced.returncode}); nothing queued")
    for spec in specs:
        planned = ssh.call(head, in_checkout(testbed, dry_run_command(spec)), timeout=300)
        problems = [line.strip() for line in (planned.stdout + planned.stderr).splitlines()
                    if line.strip().startswith("problem:") or "Traceback" in line]
        if planned.returncode or problems:
            raise Refused(f"{spec}: dry run refused (exit {planned.returncode}):\n  " + "\n  ".join(problems)
                          + ("" if problems else "\n  " + (planned.stdout + planned.stderr).strip()[-600:]))
        say(f"  dry run {spec}: no problems")
    add_all()
    again = read_state(ssh, testbed)
    if again.hold_is_queue:
        say(f"  window open ({again.hold.get('holder')}): its queue runner takes the specs")
    elif again.watchers:
        say(f"  an idle watcher is already running: {again.watchers[-1]}")
    else:
        started = ssh.call(head, in_checkout(testbed, watcher_command(idle_minutes, log_name)), retry=False)
        if started.returncode:
            raise Refused(f"{head}: the idle watcher did not start (exit {started.returncode})")
        say(f"  idle watcher started: {idle_minutes} idle minutes, then the queue runner logs to "
            f"{LAB}/{log_name}-window.log")
    return 0


def command_queue(args: argparse.Namespace, ssh: Ssh | None = None, local: Local | None = None) -> int:
    return queue_specs(ssh or Ssh(), local or Local(), testbeds_of([args.testbed])[0], args.specs,
                       apply=args.apply, idle_minutes=args.idle_minutes, log_name=args.log_name)


# ---------------------------------------------------------------- ship

def image_id_command(image: str) -> str:
    return f"docker image inspect --format '{{{{.Id}}}}' {shlex.quote(image)}"


def image_id(ssh: Ssh, host: str, image: str) -> str | None:
    result = ssh.call(host, image_id_command(image))
    found = re.search(r"sha256:[0-9a-f]{64}", result.stdout)
    return found.group(0) if result.returncode == 0 and found else None


def image_ids(ssh: Ssh, hosts: Iterable[str], image: str) -> dict[str, str | None]:
    return {host: image_id(ssh, host, image) for host in hosts}


def default_profile_dir(image: str) -> str:
    match = re.fullmatch(r"spark-ds41f:(candidate-\d+)", image)
    if not match:
        raise Refused(f"cannot derive the candidate folder from {image}; pass --profile-dir")
    return f"{CANDIDATES}/{match.group(1)}"


def parse_queue_options(values: list[str] | None, targets: list[str]) -> dict[str, list[str]]:
    queued: dict[str, list[str]] = {}
    for value in values or []:
        name, _, specs = value.partition("=")
        if name not in targets or not specs:
            raise Refused(f"--queue {value!r}: expected TESTBED=SPEC[,SPEC...] for one of {', '.join(targets)}")
        queued.setdefault(name, []).extend(spec for spec in specs.split(",") if spec)
    return queued


class Lock:
    """One image stream through the laptop at a time (two parallel streams each crawl)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            say(f"another image stream holds {self.path}; waiting for it")
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def wait_until(check: Callable[[], object], what: str, timeout: float, interval: float,
               sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
    deadline = clock() + timeout
    announced = False
    while True:
        try:
            value = check()
        except RemoteError as error:
            warn(f"{what}: {error}; checking again")
            value = None
        if value:
            return value
        if clock() >= deadline:
            raise Refused(f"timed out waiting for {what}")
        if not announced:
            say(f"waiting for {what} (every {interval:g} s, up to {timeout:g} s)")
            announced = True
        sleep(interval)


def command_ship(args: argparse.Namespace, ssh: Ssh | None = None, local: Local | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    ssh, local = ssh or Ssh(), local or Local()
    image, source = args.image, args.source
    targets = list(dict.fromkeys(args.to))
    testbeds = testbeds_of(targets)
    if any(testbed.name == "tp4-ring" for testbed in testbeds):
        raise Refused("tp4-ring gets images from the lab's image job along the ring, not from ship")
    profile_dir = repo_relative(args.profile_dir) if args.profile_dir else default_profile_dir(image)
    queued = parse_queue_options(args.queue, targets)
    heads = list(dict.fromkeys(testbed.head for testbed in testbeds))
    first = heads[0]
    load = "docker load"
    save = f"docker save {shlex.quote(image)}"
    relay = lambda target: (f"set -o pipefail; docker save {shlex.quote(image)} | ssh {' '.join(SSH_OPTIONS)} "
                            f"{shlex.quote(target)} docker load")
    say(f"ship {image} from {source} to {', '.join(targets)} (profiles in {profile_dir})")
    say("every remote command runs under bash -lc on its node")
    local.fetch()
    read = local.reader(local.rev_parse("origin/main"))
    for testbed in testbeds:
        profile = f"{profile_dir}/{testbed.profile}"
        try:
            resolved = resolve_profile(read, profile)
        except ValueError as error:
            raise Refused(f"{profile} at origin/main: {error}") from None
        if resolved.get("container", {}).get("image") != image:
            raise Refused(f"{profile} at origin/main serves {resolved.get('container', {}).get('image')}, not {image}")

    if not args.apply:
        source_id = image_id(ssh, source, image)
        say(f"  {source}: {image} {source_id or 'not present yet (ship waits for it)'}")
        ids = image_ids(ssh, heads, image)
        say(f"  laptop: ssh {source} {save} | ssh {first} {load}"
            + ("   (skipped: same ID)" if source_id and ids[first] == source_id else ""))
        for head in heads[1:]:
            say(f"  {first}$ {relay(head)}" + ("   (skipped: same ID)" if source_id and ids[head] == source_id else ""))
        for testbed in testbeds:
            profile = f"{profile_dir}/{testbed.profile}"
            say(f"  {testbed.name}: wait until its queue runs nothing, then")
            say(f"    {testbed.head}$ {in_checkout(testbed, fetch_detach_command())}")
            say(f"    {testbed.head}$ {in_checkout(testbed, sync_command())}")
            say(f"    {testbed.head}$ {in_checkout(testbed, f'bin/spark --cluster-config {shlex.quote(profile)} build distribute --apply')}")
            say(f"    verify {image} on {', '.join(testbed.nodes)} has the source's ID")
            for spec in queued.get(testbed.name, []):
                say(f"    then queue {spec} (candidate queue {testbed.name} ... --apply)")
        say("plan only; pass --apply to ship")
        return 0

    source_id = wait_until(lambda: image_id(ssh, source, image), f"{image} on {source}",
                           args.wait_timeout, args.interval, sleep=sleep)
    say(f"{source}: {image} {source_id}")

    def copy(target: str, transfer: Callable[[], tuple[bool, str]]) -> None:
        for attempt in range(1, args.attempts + 1):
            if image_id(ssh, target, image) == source_id:
                say(f"{target}: already holds {image} {source_id[:19]}; skipped")
                return
            ok, message = transfer()
            say(f"{target}: {message.splitlines()[-1] if message else ('loaded' if ok else 'failed')}")
            if image_id(ssh, target, image) == source_id:
                return
            warn(f"{target}: image ID differs after attempt {attempt}/{args.attempts}")
            sleep(args.interval)
        raise Refused(f"{target}: {image} did not arrive with ID {source_id}")

    if image_id(ssh, first, image) != source_id:
        with Lock(ROOT / RESULTS / "candidate-ship.lock"):
            copy(first, lambda: ssh.pipe(source, save, first, load, timeout=args.stream_timeout))
    else:
        say(f"{first}: already holds {image} {source_id[:19]}; skipped")
    def relayed(target: str) -> tuple[bool, str]:
        try:
            result = ssh.call(first, relay(target), timeout=args.stream_timeout, retry=False)
        except RemoteError as error:
            return False, str(error)
        return result.returncode == 0, (result.stdout + result.stderr).strip()

    for head in heads[1:]:
        copy(head, lambda head=head: relayed(head))

    for testbed in testbeds:
        profile = f"{profile_dir}/{testbed.profile}"
        ids = image_ids(ssh, testbed.nodes, image)
        if all(found == source_id for found in ids.values()):
            say(f"{testbed.name}: every node already holds {source_id[:19]}; no distribute")
        else:
            def idle() -> QueueState | None:
                state = read_state(ssh, testbed)
                if state.foreign_hold:
                    raise Refused(f"{testbed.name}: held by {state.hold.get('holder')!r}; not distributing")
                return state if not state.running else None
            wait_until(idle, f"{testbed.name}'s queue to run nothing", args.wait_timeout, args.interval, sleep=sleep)
            detached = ssh.call(testbed.head, in_checkout(testbed, fetch_detach_command()), timeout=300)
            commit = last_commit(detached.stdout)
            if detached.returncode or not commit:
                raise Refused(f"{testbed.head}: fetch and detach failed: {(detached.stdout + detached.stderr).strip()[-400:]}")
            say(f"{testbed.head}: detached at {commit[:12]}")
            synced = ssh.call(testbed.head, in_checkout(testbed, sync_command()), timeout=1800)
            say("  " + (synced.stdout + synced.stderr).strip().replace("\n", "\n  "))
            if synced.returncode:
                raise Refused(f"{testbed.name}: cluster sync failed (exit {synced.returncode})")
            distribute = f"bin/spark --cluster-config {shlex.quote(profile)} build distribute --apply"
            done = ssh.call(testbed.head, in_checkout(testbed, distribute), timeout=args.stream_timeout)
            say("  " + (done.stdout + done.stderr).strip().replace("\n", "\n  "))
            if done.returncode:
                raise Refused(f"{testbed.name}: build distribute failed (exit {done.returncode})")
            ids = image_ids(ssh, testbed.nodes, image)
        for node, found in ids.items():
            say(f"  {node}: {found or 'missing'}{'' if found == source_id else '  DIFFERS'}")
        if any(found != source_id for found in ids.values()):
            raise Refused(f"{testbed.name}: not every node holds {image} {source_id}")
        if queued.get(testbed.name):
            queue_specs(ssh, local, testbed, queued[testbed.name], apply=True,
                        idle_minutes=args.idle_minutes, log_name=args.log_name)
    return 0


# ---------------------------------------------------------------- gate

def determinism_script(run: str) -> str:
    directory = f"{RESULTS}/determinism/{run}"
    return (f"cd {shlex.quote(directory)} 2>/dev/null || exit 4\n"
            'for f in determinism-*.jsonl greedy-*.jsonl; do [ -f "$f" ] || continue; '
            'echo "@@file $f"; cat -- "$f"; echo; echo "@@eof"; done')


def parse_determinism(text: str) -> dict:
    """One token_determinism_text.py output: its scenario records and summary."""
    scenarios, summary = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "scenario" in record:
            scenarios.append(record)
        elif "summary" in record:
            summary = record
    return {"scenarios": scenarios, "summary": summary}


def gate_hashes(summary: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    """((log-probability reference, long), (text reference, long)). A text run's own hashes are
    the text hashes; only a log-probability run has log-probability hashes."""
    text = (summary.get("reference_text_sha256", ""), summary.get("long_text_sha256", ""))
    if summary.get("logprobs"):
        return (summary.get("reference_sha256", ""), summary.get("long_sha256", "")), text
    return ("", ""), (text[0] or summary.get("reference_sha256", ""), text[1] or summary.get("long_sha256", ""))


def hash_pair(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    first, slash, second = value.partition("/")
    if not slash or not re.fullmatch(r"[0-9a-fA-F]{6,64}", first) or not re.fullmatch(r"[0-9a-fA-F]{6,64}", second):
        raise Refused(f"expected HASH/HASH (hex), got {value!r}")
    return first.lower(), second.lower()


def gate_report(files: dict[str, str], expect: tuple[str, str] | None = None,
                expect_text: tuple[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(report lines, failures) for determinism-*/greedy-* outputs."""
    lines, failures = [], []
    logprob_runs = text_runs = 0
    for name in sorted(files):
        parsed = parse_determinism(files[name])
        summary = parsed["summary"]
        if summary is None:
            lines.append(f"{name}: no summary (the run did not finish)")
            failures.append(f"{name}: no summary")
            continue
        logprobs = bool(summary.get("logprobs"))
        (reference, long), (text_reference, text_long) = gate_hashes(summary)
        identical = summary.get("identical") is True
        kind = "log-probability run" if logprobs else "text run (no log probabilities)"
        lines.append(f"{name}: {kind}; identical: {'yes' if identical else 'NO'}")
        if logprobs:
            logprob_runs += 1
            lines.append(f"  log-probability hashes: reference {reference[:8]}  long {long[:8]}")
        else:
            text_runs += 1
        lines.append(f"  text hashes:            reference {text_reference[:8]}  long {text_long[:8]}")
        if not identical:
            failures.append(f"{name}: not identical")
        for scenario in parsed["scenarios"]:
            tokens, logprob = scenario.get("token_diffs", 0), scenario.get("logprob_diffs", 0)
            if not tokens and not logprob:
                continue
            lines.append(f"  {scenario['scenario']}: {tokens} token diff(s)"
                         + (f", {logprob} log-probability diff(s)" if logprobs else ""))
            for request in scenario.get("requests", []):
                differs = request.get("first_token_diff") is not None or \
                    (logprobs and request.get("logprobs_equal") is False)
                if not differs:
                    continue
                where = [f"first token diff at {request.get('first_token_diff')}"]
                if logprobs:
                    where.append(f"first log-probability diff at {request.get('first_logprob_diff')}")
                elif request.get("first_char_diff") is not None:
                    where.append(f"first character diff at {request.get('first_char_diff')}")
                lines.append(f"    prompt {request.get('prompt')}: {', '.join(where)}")
        if expect and logprobs:
            if not (reference.startswith(expect[0]) and long.startswith(expect[1])):
                failures.append(f"{name}: log-probability hashes {reference[:8]}/{long[:8]}, "
                                f"expected {expect[0]}/{expect[1]}")
        if expect_text and not (text_reference.startswith(expect_text[0]) and text_long.startswith(expect_text[1])):
            failures.append(f"{name}: text hashes {text_reference[:8]}/{text_long[:8]}, "
                            f"expected {expect_text[0]}/{expect_text[1]}")
    if expect and not logprob_runs:
        failures.append("--expect names log-probability hashes, but no log-probability run was found")
    if not files:
        failures.append("no determinism-*.jsonl or greedy-*.jsonl outputs")
    return lines, failures


def command_gate(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    testbed = testbeds_of([args.testbed])[0]
    result = ssh.call(testbed.head, in_checkout(testbed, determinism_script(args.run)))
    if result.returncode == 4:
        say(f"{testbed.head}: no {RESULTS}/determinism/{args.run}")
        return 1
    files = {name: text or "" for name, text in parse_node_files(result.stdout).items()}
    lines, failures = gate_report(files, hash_pair(args.expect), hash_pair(args.expect_text))
    say(f"gate {args.run} on {testbed.name} ({testbed.head})")
    for line in lines:
        say(line)
    for failure in failures:
        say(f"FAIL: {failure}")
    say("gate: " + ("FAIL" if failures else "pass"))
    return 1 if failures else 0


# ---------------------------------------------------------------- window-log segments

ENTRY = re.compile(r"^\d{8}T\d{6}-.+\.json$")


def entries_script(reference: str) -> str:
    """Queue entries matching a run (its spec's "run") or an entry's file name, with their specs."""
    if ENTRY.match(reference):
        test = f'[ "${{f##*/}}" = {shlex.quote(reference)} ]'
    else:
        needle = '"run": ' + json.dumps(reference)  # as queue add writes the spec (indent 2)
        test = f'grep -qF -- {shlex.quote(needle)} "$f"'
    return (f"cd {shlex.quote(QUEUE)} 2>/dev/null || exit 4\n"
            "for d in pending running done failed; do "
            'if [ "$d" = pending ]; then g=.; else g=$d; fi; '
            'for f in "$g"/*.json; do [ -f "$f" ] || continue; '
            f"if {test}; then echo \"@@entry $d ${{f##*/}}\"; cat -- \"$f\"; echo; echo \"@@eof\"; fi; "
            "done; done")


@dataclasses.dataclass
class Entry:
    state: str
    name: str
    spec: dict


def parse_entries(text: str) -> list[Entry]:
    entries, lines, index = [], text.split("\n"), 0
    while index < len(lines):
        if lines[index].startswith("@@entry "):
            state, _, name = lines[index][len("@@entry "):].partition(" ")
            body = []
            index += 1
            while index < len(lines) and lines[index] != "@@eof":
                body.append(lines[index])
                index += 1
            try:
                spec = json.loads("\n".join(body))
            except ValueError:
                spec = {}
            entries.append(Entry(state, name.strip(), spec))
        index += 1
    return entries


def find_entries(ssh: Ssh, testbed: Testbed, reference: str) -> list[Entry]:
    result = ssh.call(testbed.head, in_checkout(testbed, entries_script(reference)))
    if result.returncode == 4:
        return []
    entries = parse_entries(result.stdout)
    if not ENTRY.match(reference):
        entries = [entry for entry in entries if entry.spec.get("run") == reference]
    return sorted(entries, key=lambda entry: entry.name)


def segment_script(entry: str, body: bool) -> str:
    """Each window-log segment of a queue entry: from its first "QUEUE job X started" to
    "QUEUE job X done|failed" (a requeue restarts inside one segment)."""
    awk = ('!on && index($0, s) {on=1; start=FNR; print "@@segment " FILENAME " " FNR} '
           + ('on {print} ' if body else '')
           + 'on && (index($0, d) || index($0, f)) {on=0; print "@@segment-end " FNR} '
           'END {if (on) print "@@segment-open " FNR}')  # one file per awk: mawk has no ENDFILE
    needle = f"QUEUE job {entry} "
    return (f"cd {shlex.quote(LAB)} 2>/dev/null || exit 4\n"
            f"for lf in *.log; do grep -qF -- {shlex.quote(needle + 'started')} \"$lf\" || continue; "
            f"awk -v s={shlex.quote(needle + 'started')} -v d={shlex.quote(needle + 'done')} "
            f"-v f={shlex.quote(needle + 'failed')} {shlex.quote(awk)} \"$lf\"; done")


@dataclasses.dataclass
class Segment:
    log: str
    start: int
    end: int | None  # None while the job still runs
    lines: list[str]


def parse_segments(text: str) -> list[Segment]:
    segments, current = [], None
    for line in text.split("\n"):
        if line.startswith("@@segment "):
            log, _, start = line[len("@@segment "):].rpartition(" ")
            current = Segment(log, int(start), None, [])
            segments.append(current)
        elif line.startswith("@@segment-end ") and current:
            current.end = int(line.split()[1])
            current = None
        elif line.startswith("@@segment-open ") and current:
            current = None
        elif current is not None:
            current.lines.append(line)
    return segments


def cut_segments(text: str, entry: str) -> list[Segment]:
    """The same cut as segment_script, on a log's text (tests and local logs)."""
    started, done, failed = (f"QUEUE job {entry} {word}" for word in ("started", "done", "failed"))
    segments, current = [], None
    for number, line in enumerate(text.split("\n"), 1):
        if current is None and started in line:
            current = Segment("<text>", number, None, [])
            segments.append(current)
        if current is not None:
            current.lines.append(line)
            if done in line or failed in line:
                current.end = number
                current = None
    return segments


def read_segments(ssh: Ssh, testbed: Testbed, entry: str, body: bool = True) -> list[Segment]:
    result = ssh.call(testbed.head, in_checkout(testbed, segment_script(entry, body)))
    if result.returncode == 4:
        return []
    return parse_segments(result.stdout)


# ---------------------------------------------------------------- verdicts

def verdict_blocks(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Each table's verdict block (from "verdicts:" to the next blank line) and the suite it
    judges (the "suite NAME (" line that follows it, before the next block)."""
    blocks = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("verdicts:"):
            index += 1
            continue
        block = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip():
            block.append(lines[index])
            index += 1
        suite = None
        for later in lines[index:]:
            if later.startswith("verdicts:"):
                break
            match = re.match(r"^suite (\S+) \(", later)
            if match:
                suite = match.group(1)
                break
        blocks.append((suite, block))
    return blocks


def summary_record(line: str) -> tuple[str, dict] | None:
    """A JSON summary line, bare or prefixed "<output stem>: " (the table step's summaries)."""
    brace = line.find("{")
    if brace < 0:
        return None
    prefix = line[:brace].strip()
    if prefix and (not prefix.endswith(":") or " " in prefix):
        return None
    try:
        record = json.loads(line[brace:])
    except ValueError:
        return None
    if not isinstance(record, dict) or "summary" not in record:
        return None
    return prefix.removesuffix(":"), record


BOOT = re.compile(r"^\S+ boot (\S+): ready \+(\S+) s, command (\d+) s( FAILED)?")
# lab.describe_boot_probe's line after a boot: where it was the image's first boot, and the
# kernel-cache files it added per node.
BOOT_PROBE = re.compile(r"^\S+ boot (\S+): (?:first boot of the image on (.+?)|not the image's first boot on any node); "
                        r"kernel cache (.+) files")
SUITE_STEP = re.compile(r"^servebench suite (\S+): step ")
EXIT = re.compile(r"^\S+Z (\S+) (\S+) exit (-?\d+)$")


@dataclasses.dataclass
class ArmNumbers:
    label: str
    boots: list = dataclasses.field(default_factory=list)  # (config, ready s, command s, failed)
    # Per boot, parallel to boots: (first-boot nodes, kernel-cache files added) from the lab's
    # probe line, or None for a log written before the probe.
    probes: list = dataclasses.field(default_factory=list)
    closed: dict = dataclasses.field(default_factory=dict)  # (suite run, concurrency) -> (tps, ci95 %)
    prefill: dict = dataclasses.field(default_factory=dict)  # (suite run, tokens) -> tps
    model: dict = dataclasses.field(default_factory=dict)  # suite run -> record
    distinct: dict = dataclasses.field(default_factory=dict)  # output stem -> (streams, tps, ci95 %)
    exits: list = dataclasses.field(default_factory=list)  # (step, code, suite or None)


def spec_labels(spec: dict) -> list[str]:
    labels = []
    for job in spec.get("jobs", []):
        arms = job.get("arms", []) + ([job["arm"]] if isinstance(job.get("arm"), dict) else [])
        labels += [arm["label"] for arm in arms if "label" in arm]
        if job.get("bracket") and job.get("arms") and "suite" not in job:
            labels.append(f"{job['arms'][0]['label']}-end")
    return list(dict.fromkeys(labels))


def headline_numbers(lines: list[str], labels: list[str]) -> dict[str, ArmNumbers]:
    """Each arm's headline numbers from a window-log segment, in the order arms appear."""
    arms: dict[str, ArmNumbers] = {}
    pending_boot = None
    pending_probe = None
    suite = None  # the suite whose steps the log shows now

    def arm(label: str) -> ArmNumbers:
        nonlocal pending_boot, pending_probe
        if label not in arms:
            arms[label] = ArmNumbers(label)
        if pending_boot:
            arms[label].boots.append(pending_boot)
            arms[label].probes.append(pending_probe)
            pending_boot = pending_probe = None
        return arms[label]

    def from_output(record: dict, suffix: str) -> tuple[str, str]:
        output = record.get("output")
        if output:
            path = Path(output)
            return path.parent.name, path.parent.parent.name
        return str(record.get("label", "")).removesuffix(suffix), ""

    for line in lines:
        stepping = SUITE_STEP.match(line)
        if stepping:
            suite = stepping.group(1)
            continue
        booted = BOOT.match(line)
        if booted:
            suite = None
            ready = None if booted.group(2) == "None" else float(booted.group(2))
            pending_boot = (booted.group(1), ready, int(booted.group(3)), bool(booted.group(4)))
            pending_probe = None
            continue
        probed = BOOT_PROBE.match(line)
        if probed and pending_boot:
            files = {}
            for part in probed.group(3).split(", "):
                node, _, count = part.partition(" +")
                if count.isdigit():
                    files[node] = int(count)
            pending_probe = ([n for n in (probed.group(2) or "").split(", ") if n], files)
            continue
        exited = EXIT.match(line)
        if exited and exited.group(2) in labels or (exited and not labels and exited.group(1) == "bench"):
            arm(exited.group(2)).exits.append((exited.group(1), int(exited.group(3)),
                                               suite if exited.group(1) == "bench" else None))
            suite = None
            continue
        found = summary_record(line)
        if not found:
            continue
        prefix, record = found
        kind = record["summary"]
        if kind == "servebench closed":
            label, run = from_output(record, f"-c{record.get('concurrency')}")
            arm(label).closed[(run, record.get("concurrency"))] = (record.get("tps_mean"), record.get("ci95_pct"))
        elif kind == "servebench prefill":
            label, run = from_output(record, "-prefill")
            for tokens, tps in (record.get("tps") or {}).items():
                arm(label).prefill[(run, int(tokens))] = tps
        elif kind == "servebench model":
            label, run = from_output(record, "-model")
            arm(label).model[run] = record
        elif kind.endswith("distinct prompts") and prefix:
            label = next((l for l in sorted(labels, key=len, reverse=True) if prefix.endswith(f"-{l}")), None)
            stem = prefix[:-len(label) - 1] if label else prefix
            arm(label or prefix).distinct[stem] = (record.get("streams"), record.get("tps_mean"), record.get("ci95_pct"))
    return arms


def cold_boot(probe: tuple | None) -> bool | None:
    """A boot to leave out of time-to-ready comparisons: one that compiled kernels (an image's
    first boot on a node does). The ledger's first-boot mark decides only when no kernel cache
    could be read: a ledger started after an image's first boot marks its next boot first too.
    None when the log predates the probe."""
    if probe is None:
        return None
    first, files = probe
    return any(files.values()) if files else bool(first)


def ready_comparison(arms: dict[str, ArmNumbers]) -> list[str]:
    """Time to ready per arm over its warm boots only: an image's first boot on a node
    compiles its changed kernels (tens of seconds to minutes), which is not the arm's cost."""
    lines, left_out, unknown = [], [], 0
    for numbers in arms.values():
        warm = []
        for index, (config, ready, command, failed) in enumerate(numbers.boots):
            probe = numbers.probes[index] if index < len(numbers.probes) else None
            if ready is None or failed:
                continue
            cold = cold_boot(probe)
            if cold:
                left_out.append(f"{numbers.label} ({config}, +{ready} s)")
                continue
            unknown += cold is None
            warm.append(ready)
        if warm:
            lines.append(f"  {numbers.label}: {sum(warm) / len(warm):.1f} s over {len(warm)} warm boot"
                         f"{'s' if len(warm) > 1 else ''}")
    if not lines and not left_out:
        return []
    head = ["time to ready (warm boots; an image's first boot on a node, which compiles kernels, is left out):"]
    tail = ([f"  left out: {'; '.join(left_out)}"] if left_out else [])
    tail += ([f"  {unknown} boot{'s' if unknown > 1 else ''} without a kernel-cache record (a log before the probe) "
              "counted as warm"] if unknown else [])
    return head + lines + tail


def describe_arm(numbers: ArmNumbers, run: str) -> list[str]:
    lines = [f"  {numbers.label}"]
    for index, (config, ready, command, failed) in enumerate(numbers.boots):
        probe = numbers.probes[index] if index < len(numbers.probes) else None
        note = ""
        if probe is not None:
            first, files = probe
            compiled = sum(files.values())
            if cold_boot(probe):
                note = (f"; {'first boot of the image on ' + ', '.join(first) if first else 'kernels compiled'}"
                        f", kernel cache +{compiled} files: left out of time-to-ready comparisons")
            else:
                note = "; warm (no kernel compiled)"
        lines.append(f"    boot {config}: ready +{ready if ready is not None else '?'} s "
                     f"(command {command} s){' FAILED' if failed else ''}{note}")
    suite_of = lambda suite_run: "" if suite_run in ("", run) else suite_run.removeprefix(f"{run}-") + " "
    for (suite_run, concurrency), (tps, ci) in sorted(numbers.closed.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0))):
        lines.append(f"    {suite_of(suite_run)}closed loop c{concurrency}: {tps:.2f} tok/s"
                     + (f" ±{ci:.2f}%" if isinstance(ci, (int, float)) else ""))
    by_run: dict[str, list] = {}
    for (suite_run, tokens), tps in sorted(numbers.prefill.items()):
        by_run.setdefault(suite_run, []).append(f"{tokens:,} {tps:,.0f} tok/s")
    for suite_run, values in by_run.items():
        lines.append(f"    {suite_of(suite_run)}prefill: {'; '.join(values)}")
    for suite_run, record in numbers.model.items():
        accept, perplexity = record.get("acceptance_length"), record.get("perplexity")
        number = lambda value, digits: f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"
        lines.append(f"    {suite_of(suite_run)}model: acceptance length {number(accept, 3)}, "
                     f"perplexity {number(perplexity, 5)}")
    for stem, (streams, tps, ci) in numbers.distinct.items():
        lines.append(f"    {stem}: {tps} tok/s" + (f" ±{ci}%" if ci is not None else "") + f" at {streams} streams")
    for step, code, suite in numbers.exits:
        if code:
            lines.append(f"    {step}{f' --suite {suite}' if suite else ''} exit {code}")
    return lines


def tracebacks(lines: list[str]) -> list[tuple[int, str]]:
    """(line number in the segment, the exception it ended with) for each traceback."""
    found = []
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            ending = next((later for later in lines[index + 1:index + 200]
                           if later and not later.startswith((" ", "\t"))), "?")
            found.append((index + 1, ending))
    return found


def resolve_entries(ssh: Ssh, testbed: Testbed, reference: str) -> list[Entry]:
    entries = find_entries(ssh, testbed, reference)
    if not entries:
        raise Refused(f"{testbed.name}: no queue entry for {reference!r} (pending, running, done or failed)")
    return entries


def command_verdicts(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    testbed = testbeds_of([args.testbed])[0]
    entries = resolve_entries(ssh, testbed, args.reference)
    for entry in entries:
        run = entry.spec.get("run", "?")
        segments = read_segments(ssh, testbed, entry.name)
        say(f"== {entry.name} ({entry.state}), run {run}")
        if not segments:
            say(f"  no window-log segment under {LAB}/*.log (was it run by the queue?)")
            continue
        lines = [line for segment in segments for line in segment.lines]
        for segment in segments:
            say(f"  window log {segment.log} lines {segment.start}-{segment.end or '(still running)'}")
        for number, ending in tracebacks(lines):
            say(f"  traceback at segment line {number}: {ending[:200]}")
        blocks = verdict_blocks(lines)
        if not blocks:
            say("  no verdict block (a job without a suite table, or one that has not reached its table)")
        for suite, block in blocks:
            say("")
            say(f"[{suite or 'table'}]")
            for line in block:
                say(line)
        say("")
        say("arms:")
        arms = headline_numbers(lines, spec_labels(entry.spec))
        for numbers in arms.values():
            for line in describe_arm(numbers, run):
                say(line)
        for line in ready_comparison(arms):
            say(line)
        say("")
    return 0


# ---------------------------------------------------------------- receipts

def spec_runs_of(spec: dict) -> list[str]:
    """The run and the runs of its companion suites (``<run>-<suite>``)."""
    run = spec.get("run", "")
    runs = [run]
    for job in spec.get("jobs", []):
        suites = job.get("suite")
        if isinstance(suites, list):
            runs += [f"{run}-{suite}" for suite in suites[1:]]
    return list(dict.fromkeys(runs))


def receipt_sources(spec: dict) -> list[str]:
    """Paths under results/private that hold a run's evidence."""
    run = spec["run"]
    runs = spec_runs_of(spec)
    return ([f"lab/{name}-table.txt" for name in runs] + [f"lab/overlays-{run}.txt", f"lab/{run}",
            f"determinism/{run}"] + [f"suites/{name}" for name in runs])


def listing_script(paths: list[str]) -> str:
    quoted = " ".join(shlex.quote(path) for path in paths)
    return (f"cd {shlex.quote(RESULTS)} 2>/dev/null || exit 4\n"
            f"for p in {quoted}; do if [ -e \"$p\" ]; then find \"$p\" -type f -printf '%s %p\\n'; "
            "else echo \"@@missing $p\"; fi; done")


def classify(path: str) -> str:
    """copy, large (left on the node, listed in the TSV) or skip."""
    if path.startswith("suites/"):
        name = path.rsplit("/", 1)[-1]
        if name == "model.json.gz":
            return "large"
        if fnmatch.fnmatch(name, "*.engine-*.log"):
            return "skip"
    return "copy"


def tar_command(paths: list[str]) -> str:
    quoted = " ".join(shlex.quote(path) for path in paths)
    return (f"cd {shlex.quote(RESULTS)} && tar -cf - --exclude='suites/*/model.json.gz' "
            f"--exclude='suites/*.engine-*.log' -- {quoted}")


def receipts_destination(dest: str) -> Path:
    relative = repo_relative(dest)
    parts = Path(relative).parts
    if not parts or parts[0] != "experiments" or "receipts" not in parts:
        raise Refused(f"{dest}: receipts go under an experiment's receipts/ folder")
    return ROOT / relative


def human(size: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return str(size)


TSV_HEADER = "sha256\tbytes\tlocation"


def command_receipts(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    testbed = testbeds_of([args.testbed])[0]
    dest = receipts_destination(args.dest)
    entries = resolve_entries(ssh, testbed, args.reference)
    runs = list(dict.fromkeys(entry.spec.get("run") for entry in entries))
    if len(runs) != 1 or not runs[0]:
        raise Refused(f"entries {', '.join(e.name for e in entries)} name runs {runs}; give one queue entry")
    run = runs[0]
    spec = entries[-1].spec
    unfinished = [entry for entry in entries if entry.state in ("pending", "running")]
    sources = receipt_sources(spec)
    listing = ssh.call(testbed.head, in_checkout(testbed, listing_script(sources)))
    if listing.returncode:
        raise RemoteError(f"{testbed.head}: cannot list {RESULTS} (exit {listing.returncode})")
    files, missing = [], []
    for line in listing.stdout.splitlines():
        if line.startswith("@@missing "):
            missing.append(line[len("@@missing "):])
        elif " " in line and line.split(" ", 1)[0].isdigit():
            size, path = line.split(" ", 1)
            files.append((path, int(size)))
    copies = [(path, size) for path, size in files if classify(path) == "copy"]
    large = [(path, size) for path, size in files if classify(path) == "large"]
    skipped = [path for path, _ in files if classify(path) == "skip"]
    segments = [segment for entry in entries for segment in read_segments(ssh, testbed, entry.name, body=False)]
    window_log = f"lab/{run}-window.log"
    targets = [path for path, _ in copies] + ([window_log] if segments else [])
    conflicts = [path for path in targets if (dest / path).exists()]
    tsv = dest / "large-files-on-nodes.tsv"

    say(f"receipts for {run} on {testbed.name} ({testbed.head}) -> {dest.relative_to(ROOT)}")
    for entry in entries:
        say(f"  queue entry: {entry.state}/{entry.name}")
    for segment in segments:
        say(f"  window log: {segment.log} lines {segment.start}-{segment.end or '(running)'} -> {window_log}")
    if not segments:
        say(f"  window log: no segment found under {LAB}/*.log")
    groups: dict[str, list[int]] = {}
    for path, size in copies:
        parts = path.split("/")
        group = path if parts[0] == "lab" and len(parts) == 2 else "/".join(parts[:2]) + "/"
        groups.setdefault(group, []).append(size)
    for group, sizes in groups.items():
        say(f"  copy  {group}" + (f" ({len(sizes)} files, {human(sum(sizes))})" if group.endswith("/")
                                  else f" ({human(sizes[0])})"))
    for path, size in large:
        say(f"  large {path} ({human(size)}): left on {testbed.head}, listed in {tsv.name}")
    if skipped:
        say(f"  skip  {len(skipped)} engine log(s) under suites/")
    for path in missing:
        say(f"  (none on the node: {path})")
    say(f"  conflicts in the destination: {', '.join(conflicts) if conflicts else 'none'}")
    if unfinished:
        say(f"  NOTE: {', '.join(e.name for e in unfinished)} has not finished; receipts are final evidence")
    if not args.apply:
        say("plan only; pass --apply to copy")
        return 0
    if unfinished:
        raise Refused("the run has not finished; copy its receipts once it is done or failed")
    if conflicts and not args.force:
        raise Refused(f"{len(conflicts)} file(s) exist in {dest.relative_to(ROOT)}; pass --force to overwrite")
    if tsv.exists() and not tsv.read_text(encoding="utf-8").startswith(TSV_HEADER):
        raise Refused(f"{tsv.relative_to(ROOT)} has another header; add its rows by hand")

    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="receipts-", dir=work))
    try:
        present = [path for path in sources if path not in missing]

        def extract(stream) -> None:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                archive.extractall(staging, filter="data")

        if present:
            ssh.stream(testbed.head, in_checkout(testbed, tar_command(present)), extract, timeout=3600)
        got = sorted(str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file())
        expected = sorted(path for path, _ in copies)
        if got != expected:
            extra, lost = sorted(set(got) - set(expected)), sorted(set(expected) - set(got))
            raise Refused(f"the copy differs from the listing (extra {extra[:5]}, missing {lost[:5]}); retry")
        if segments:
            full = [segment for entry in entries for segment in read_segments(ssh, testbed, entry.name)]
            (staging / "lab").mkdir(exist_ok=True)
            (staging / window_log).write_text("".join("\n".join(s.lines) + "\n" for s in full), encoding="utf-8")
        rows = []
        if large:
            sums = ssh.call(testbed.head, in_checkout(
                testbed, f"cd {shlex.quote(RESULTS)} && sha256sum -- " + " ".join(shlex.quote(p) for p, _ in large)))
            digests = {line.split()[1]: line.split()[0] for line in sums.stdout.splitlines()
                       if len(line.split()) == 2 and re.fullmatch(r"[0-9a-f]{64}", line.split()[0])}
            for path, size in large:
                if path not in digests:
                    raise Refused(f"{testbed.head}: no sha256 for {path}")
                rows.append(f"{digests[path]}\t{size}\t{testbed.head}:{testbed.checkout}/{RESULTS}/{path}")
        moved = 0
        for path in sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()):
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / path, target)
            moved += 1
        if rows:
            existing = tsv.read_text(encoding="utf-8").splitlines() if tsv.exists() else [TSV_HEADER]
            added = [row for row in rows if row not in existing]
            tsv.parent.mkdir(parents=True, exist_ok=True)
            tsv.write_text("\n".join(existing + added) + "\n", encoding="utf-8")
            say(f"  {tsv.relative_to(ROOT)}: {len(added)} row(s) added")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logs = [path for path in targets if path.endswith(".log")]
    say(f"copied {moved} file(s) into {dest.relative_to(ROOT)}")
    if logs:
        say(f"  .gitignore ignores *.log: add {len(logs)} log(s) with git add -f")
    return 0


# ---------------------------------------------------------------- overlay

def overlay_files_script(tree: str, commit: str) -> str:
    git = f"git -C {home_path(tree)}"
    quoted = shlex.quote(commit)
    return "\n".join([
        f"{git} rev-parse --verify -q {shlex.quote(commit + '^{commit}')} >/dev/null || exit 4",
        "echo @@section parents", f"{git} rev-list --parents -n 1 {quoted}",
        "echo @@section subject", f"{git} log -1 --format=%s {quoted}",
        "echo @@section changes", f"{git} diff-tree --no-commit-id -r --name-status {quoted}",
    ])


def later_commits_script(tree: str, commit: str, files: list[str]) -> str:
    return (f"git -C {home_path(tree)} log --format='%H %s' {shlex.quote(commit + '..HEAD')} -- "
            + " ".join(shlex.quote(path) for path in files))


def archive_command(tree: str, commit: str, files: list[str]) -> str:
    return (f"git -C {home_path(tree)} archive --format=tar {shlex.quote(commit + '^')} -- "
            + " ".join(shlex.quote(path) for path in files))


def overlay_plan(changes: list[str]) -> tuple[list[str], list[str]]:
    """(files to restore, refusals) from ``git diff-tree --name-status`` lines, tests/ excluded."""
    files, refusals = [], []
    for line in changes:
        if not line.strip():
            continue
        status, *paths = line.split("\t")
        path = paths[-1] if paths else ""
        if path.startswith("tests/") or (paths and paths[0].startswith("tests/")):
            continue
        if status == "M":
            files.append(path)
        else:
            refusals.append(f"{status} {' -> '.join(paths)}")
    return files, refusals


def overlay_profiles(folder: str, name: str, shape: str, files: list[str], base_mounts: list,
                     later: bool, note: str | None) -> dict[str, dict]:
    candidate = Path(folder).name
    overlay = f"{folder}/{name}/overlay"
    mounts = [list(mount) for mount in base_mounts
              if not any(mount[1] == f"{SOURCE_ROOT}/vllm/{path}" for path in files)]
    mounts += [[f"{{checkout}}/{overlay}/{path}", f"{SOURCE_ROOT}/vllm/{path}", "ro"] for path in files]
    where = SHAPE_NAMES.get(shape, shape)
    notes = (f"{candidate} without {name}: the {len(files)} non-test file(s) the reverted patch changes, at "
             f"their versions before it ({'later patches touch them too, and the overlay reverts those as well' if later else 'no later patch touches them'}), "
             f"mounted from {name}/overlay over the {candidate} image. A control at {where}.")
    if note:
        notes += f" {note.strip()}"
    profile = f"{folder}/{shape}-{name}.json"
    return {
        profile: {"notes": notes, "extends": {"config": f"{folder}/{shape}.json"},
                  "set": {"container": {"mounts": mounts}}},
        f"{folder}/{shape}-{name}-dev.json": {
            "notes": f"{shape}-{name}.json with the dev-mode collective RPC.",
            "extends": {"config": profile}, "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}}},
    }


def base_mounts(folder: str, shape: str) -> list:
    """The base profile's own mounts (the cache mount, and any overlay it already has). The
    model's mounts and the GLP file's are the recipe's renderings: an arm's mount list never
    names them, and the overlay renders them again."""
    resolved = resolve_profile(Local().reader(None), f"{folder}/{shape}.json")
    return [mount for mount in resolved.get("container", {}).get("mounts", [])
            if mount[1] not in recipe.MODEL_MOUNTS and mount[1] != recipe.GLP_TARGET]


def command_overlay(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    folder = repo_relative(args.folder)
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise Refused(f"{name!r}: a control's name is a plain word such as no956")
    host, colon, tree = args.tree.partition(":")
    if not colon or not host or not tree:
        raise Refused("--tree is HOST:PATH (the prepared vLLM tree with the series applied)")
    shapes = [shape.strip() for shape in args.profiles.split(",") if shape.strip()]
    for shape in shapes:
        if not (ROOT / folder / f"{shape}.json").is_file():
            raise Refused(f"{folder}/{shape}.json does not exist")
    found = ssh.call(host, overlay_files_script(tree, args.revert))
    if found.returncode == 4:
        raise Refused(f"{host}:{tree}: no commit {args.revert}")
    if found.returncode:
        raise RemoteError(f"{host}: git failed (exit {found.returncode}): {found.stderr.strip()[-300:]}")
    parts = sections(found.stdout)
    if len((parts.get("parents") or [""])[0].split()) != 2:
        raise Refused(f"{args.revert} is not a commit with one parent")
    files, refusals = overlay_plan(parts.get("changes", []))
    say(f"revert control {name}: {args.revert} ({(parts.get('subject') or [''])[0]})")
    if refusals:
        raise Refused("an overlay can only restore files the commit modifies; it also "
                      + ", ".join(refusals) + ". Build this control by hand.")
    if not files:
        raise Refused(f"{args.revert} changes no file outside tests/: nothing to overlay")
    later = ssh.call(host, later_commits_script(tree, args.revert, files))
    if later.returncode:
        raise RemoteError(f"{host}: git log failed (exit {later.returncode})")
    later_commits = [line for line in later.stdout.splitlines() if re.match(r"^[0-9a-f]{40} ", line)]
    for path in files:
        say(f"  {path}")
    if later_commits:
        say("later commits in the tree also touch these files:")
        for line in later_commits:
            say(f"  {line[:12]} {line[41:]}")
        if not args.allow_later:
            raise Refused("the overlay holds the files as they were before the reverted commit, so it would "
                          "revert these later changes too; pass --allow-later if that is the control you want")
        say("--allow-later: the overlay reverts those later changes as well")
    overlay_dir = ROOT / folder / name / "overlay"
    profiles: dict[str, dict] = {}
    for shape in shapes:
        profiles.update(overlay_profiles(folder, name, shape, files, base_mounts(folder, shape),
                                         bool(later_commits), args.note))
    texts = {path: checked_json(path, data) for path, data in profiles.items()}
    exists = [path for path in texts if (ROOT / path).exists()] + \
        ([str(overlay_dir.relative_to(ROOT))] if (ROOT / folder / name).exists() else [])
    say(f"  {host}$ {archive_command(tree, args.revert, files)} | (extract into {folder}/{name}/overlay/)")
    for path in texts:
        say(f"  write {path}")
    if exists:
        say(f"  exist already: {', '.join(exists)}")
    if not args.apply:
        say("plan only; pass --apply to write the overlay and its profiles")
        return 0
    if exists:
        raise Refused("refusing to overwrite: " + ", ".join(exists))
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=ROOT / folder))
    try:
        def extract(stream) -> None:
            shutil.rmtree(staging / "overlay", ignore_errors=True)
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                archive.extractall(staging / "overlay", filter="data")

        ssh.stream(host, archive_command(tree, args.revert, files), extract, timeout=600)
        got = sorted(str(path.relative_to(staging / "overlay")) for path in (staging / "overlay").rglob("*")
                     if path.is_file())
        if got != sorted(files):
            raise Refused(f"the archive holds {len(got)} files, not the {len(files)} listed; nothing written")
        (ROOT / folder / name).mkdir()
        os.replace(staging / "overlay", overlay_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for path, text in texts.items():
        (ROOT / path).write_text(text, encoding="utf-8")
    problems = lint_paths(list(texts), Local().reader(None))
    for problem in problems:
        say(f"lint: {problem}")
    say(f"wrote {len(files)} overlay file(s) and {len(texts)} profile(s)")
    return 1 if problems else 0


# ---------------------------------------------------------------- close

def command_close(args: argparse.Namespace, ssh: Ssh | None = None) -> int:
    ssh = ssh or Ssh()
    testbed = testbeds_of([args.testbed])[0]
    state = read_state(ssh, testbed)
    say("\n".join(describe_state(state)))
    if state.running or state.pending:
        raise Refused(f"{testbed.name}: the queue has work ({', '.join(state.running + state.pending)}); "
                      "not closing (and never write ~/spark-request.json: the runner reads it between steps "
                      "and stops the running job)")
    if not state.hold:
        say(f"{testbed.name}: no window is open")
        return 0
    if not str(state.hold.get("holder", "")).startswith("spark-lab"):
        raise Refused(f"{testbed.name}: the hold is {state.hold.get('holder')!r}, not the lab's")
    close = "python3 scripts/lab.py window close"
    say(f"  {testbed.head}$ {in_checkout(testbed, close)}")
    say(f"  {testbed.head}$ {in_checkout(testbed, 'bin/spark doctor --live')}")
    if not args.apply:
        say("plan only; pass --apply to close the window")
        return 0
    closed = ssh.call(testbed.head, in_checkout(testbed, close), timeout=3600, retry=False)
    say((closed.stdout + closed.stderr).strip())
    doctor = ssh.call(testbed.head, in_checkout(testbed, "bin/spark doctor --live"), timeout=900)
    say((doctor.stdout + doctor.stderr).strip())
    return closed.returncode or doctor.returncode


# ---------------------------------------------------------------- lint command

def command_lint(args: argparse.Namespace) -> int:
    paths = [repo_relative(path) for path in args.files]
    problems = lint_paths(paths, Local().reader(None))
    for problem in problems:
        say(problem)
    say(f"{len(paths)} file(s): " + (f"{len(problems)} problem(s)" if problems else "clean"))
    return 1 if problems else 0


# ---------------------------------------------------------------- parser

def guarded(function: Callable) -> Callable:
    def run(args: argparse.Namespace) -> int:
        try:
            return function(args)
        except Refused as error:
            print(f"refused: {error}", file=sys.stderr, flush=True)
            return 1
        except RemoteError as error:
            print(f"ssh: {error}", file=sys.stderr, flush=True)
            return 1
    return run


def add_parser(commands) -> None:
    testbeds = ", ".join(f"{t.name} ({t.head})" for t in TESTBEDS.values())
    candidate = commands.add_parser(
        "candidate", help="the candidate process's orchestration from the owner's laptop, over ssh",
        description=f"Testbeds: {testbeds}. Commands that change anything print their plan and act "
                    "only with --apply.")
    sub = candidate.add_subparsers(dest="candidate_command", required=True)

    status = sub.add_parser("status", help="each head's hold, queue, idle watcher and serving image")
    status.add_argument("testbeds", nargs="*", metavar="testbed")
    status.set_defaults(func=guarded(command_status))

    queue = sub.add_parser("queue", help="queue specs safely: never move a checkout a job runs from")
    queue.add_argument("testbed", choices=sorted(TESTBEDS))
    queue.add_argument("specs", nargs="+", metavar="spec", help="repository-relative lab spec")
    queue.add_argument("--idle-minutes", type=int, default=10,
                       help="idle minutes the watcher waits for before the queue runner starts (default 10)")
    queue.add_argument("--log-name", help="window log results/private/lab/NAME-window.log "
                                          "(default: the candidate folder's name)")
    queue.add_argument("--apply", action="store_true")
    queue.set_defaults(func=guarded(command_queue))

    wait = sub.add_parser("wait", help="block until a queue entry finishes on any listed testbed "
                                       "(exit 0; 2 on timeout)")
    wait.add_argument("testbeds", nargs="*", metavar="testbed")
    wait.add_argument("--timeout", type=float, default=21600, help="seconds (default 21600; 0: none)")
    wait.add_argument("--interval", type=float, default=60, help="seconds between polls (default 60)")
    wait.set_defaults(func=guarded(command_wait))

    ship = sub.add_parser("ship", help="stream an image to spaceport heads, distribute it and verify IDs")
    ship.add_argument("image")
    ship.add_argument("--from", dest="source", required=True, help="host that holds the built image")
    ship.add_argument("--to", nargs="+", required=True, choices=sorted(set(TESTBEDS) - {"tp4-ring"}))
    ship.add_argument("--profile-dir", help="candidate folder (default: from the image tag)")
    ship.add_argument("--queue", action="append", metavar="TESTBED=SPEC[,SPEC]",
                      help="after a verified distribute, queue these specs there")
    ship.add_argument("--idle-minutes", type=int, default=10)
    ship.add_argument("--log-name")
    ship.add_argument("--wait-timeout", type=float, default=6 * 3600,
                      help="seconds to wait for the image on the source or an idle queue (default 6 h)")
    ship.add_argument("--stream-timeout", type=float, default=4 * 3600, help="seconds a copy may take (4 h)")
    ship.add_argument("--interval", type=float, default=60)
    ship.add_argument("--attempts", type=int, default=3, help="copy attempts per head (default 3)")
    ship.add_argument("--apply", action="store_true")
    ship.set_defaults(func=guarded(command_ship))

    gate = sub.add_parser("gate", help="the determinism gate's outputs of a run")
    gate.add_argument("testbed", choices=sorted(TESTBEDS))
    gate.add_argument("run")
    gate.add_argument("--expect", metavar="REF/LONG",
                      help="log-probability hashes (reference_sha256/long_sha256 of the log-probability run)")
    gate.add_argument("--expect-text", metavar="REF/LONG",
                      help="text hashes (reference_text_sha256/long_text_sha256)")
    gate.set_defaults(func=guarded(command_gate))

    verdicts = sub.add_parser("verdicts", help="each table's verdict block and arm headlines, from the window log")
    verdicts.add_argument("testbed", choices=sorted(TESTBEDS))
    verdicts.add_argument("reference", metavar="run|queue-entry")
    verdicts.set_defaults(func=guarded(command_verdicts))

    receipts = sub.add_parser("receipts", help="copy a run's evidence into a candidate's receipts/")
    receipts.add_argument("testbed", choices=sorted(TESTBEDS))
    receipts.add_argument("reference", metavar="run|queue-entry")
    receipts.add_argument("dest", help="repository path under a candidate's receipts/")
    receipts.add_argument("--force", action="store_true", help="overwrite existing files")
    receipts.add_argument("--apply", action="store_true")
    receipts.set_defaults(func=guarded(command_receipts))

    overlay = sub.add_parser("overlay", help="a revert control: a commit's files at their parent "
                                             "versions, mounted over the candidate image")
    overlay.add_argument("folder", help="candidate folder")
    overlay.add_argument("name", help="control name, such as no956")
    overlay.add_argument("--tree", required=True, metavar="HOST:PATH", help="prepared vLLM tree")
    overlay.add_argument("--revert", required=True, metavar="COMMIT", help="the patch's commit in that tree")
    overlay.add_argument("--profiles", default="tp4,tp4-switched,tp3-switched",
                         help="shape profiles to extend (default tp4,tp4-switched,tp3-switched)")
    overlay.add_argument("--allow-later", action="store_true",
                         help="also revert later commits that touch the same files")
    overlay.add_argument("--note", help="sentence appended to the profile notes (no hashes)")
    overlay.add_argument("--apply", action="store_true")
    overlay.set_defaults(func=guarded(command_overlay))

    close = sub.add_parser("close", help="close the lab window when its queue is empty")
    close.add_argument("testbed", choices=sorted(TESTBEDS))
    close.add_argument("--apply", action="store_true")
    close.set_defaults(func=guarded(command_close))

    lint = sub.add_parser("lint", help="lint specs, profiles and Markdown the candidate process writes")
    lint.add_argument("files", nargs="+")
    lint.set_defaults(func=guarded(command_lint))
