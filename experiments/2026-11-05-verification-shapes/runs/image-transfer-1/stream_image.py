"""Transfer one immutable image; never start a container or store an archive."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

mode = sys.argv[1]
assert mode in ("export", "import")
root = Path(__file__).resolve().parent
plan = json.loads((root / "plan.json").read_text())
tag = plan["tag"]
record_path = root / (mode + "-operation.json")
assert not record_path.exists(), "preserve the existing transfer attempt"
record = {
    "operation": mode, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tag": tag, "source_image": plan["image"], "gpu": False,
    "workstation_archive": False, "passed": False,
}
def save():
    record_path.write_text(json.dumps(record, indent=2) + "\n")
save()
try:
    current = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True)
    if mode == "export":
        assert current.returncode == 0 and json.loads(current.stdout)[0]["Id"] == plan["image"]
    else:
        assert current.returncode != 0, "destination tag already exists"
    limiter = [sys.executable, str(root / "stream_limit.py"), "--receipt",
               str(root / (mode + "-stream.json"))]
    if mode == "export":
        commands = [["docker", "image", "save", tag],
                    ["zstd", "-T1", "-3", "--no-progress", "-c"], limiter]
    else:
        commands = [limiter, ["zstd", "-d", "--no-progress", "-c"],
                    ["docker", "image", "load"]]
    record["commands"] = commands
    save()
    with (root / (mode + "-stderr.log")).open("w") as err:
        if mode == "import":
            with (root / "docker-load.log").open("w") as log:
                result = subprocess.run(["bash", "-o", "pipefail", "-c",
                    " | ".join(shlex.join(c) for c in commands)], stdout=log, stderr=err)
        else:
            result = subprocess.run(["bash", "-o", "pipefail", "-c",
                " | ".join(shlex.join(c) for c in commands)], stderr=err)
    record["exit"] = result.returncode
    assert result.returncode == 0, "transfer pipeline failed"
    inspection = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))
    (root / (mode + "-inspection.json")).write_text(json.dumps(inspection, indent=2) + "\n")
    original = json.loads((root / "built-inspection.json").read_text())[0]
    current = inspection[0]
    assert current["RootFS"]["Layers"] == original["RootFS"]["Layers"], "image layers differ"
    assert current["Config"] == original["Config"], "runtime configuration differs"
    record["destination_image" if mode == "import" else "source_image_after"] = current["Id"]
    record["passed"] = True
except BaseException as error:
    record["error"] = type(error).__name__ + ": " + str(error)
    raise
finally:
    record["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
