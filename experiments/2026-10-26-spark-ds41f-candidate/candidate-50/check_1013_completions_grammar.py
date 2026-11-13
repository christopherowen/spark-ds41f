#!/usr/bin/env python3
"""#1013 live check: a /v1/completions request with a JSON-schema grammar, served under the
deepseek_v41 reasoning parser, returns schema-valid output from its first token. A lab extra.

usage: check_1013_completions_grammar.py URL [--model NAME] [--container NAME] [--timeout S]

The fix (0770, `StructuredOutputManager._get_constraint_start`): a completions request has no
reasoning phase, so its grammar applies from token 0. On 1107 the manager built a reasoner with
empty kwargs, which defaults to thinking, never saw the reasoning end in a completions prompt,
and left every bitmask row unconstrained (candidate-48's own mark, `cs=6, UUUUUU`).

Three requests, temperature 0, to URL/v1/completions with a prose prompt:
- `grammar`, whole and `grammar-stream`, streamed: `structured_outputs: {"json": SCHEMA}`
  (`response_format` json_schema if the server refuses that field). Each must finish with
  "stop", its text must be one JSON value valid against SCHEMA, and nothing but whitespace
  may come before the opening brace. The stream's first text delta is recorded.
- `unconstrained`: the same request without a grammar. Not required: it shows 1107's
  behaviour without booting 1107. 1107 applied no grammar row to a completions request, so at
  temperature 0 it returned this text for the grammar request too. The prompt asks for prose,
  so this text is not schema-valid; if it is, the comparison shows nothing, and the summary
  says so (`shows_1107_behaviour` false).

With --container (the serving container on this host, the head), the check also requires
that the server runs with `--reasoning-parser deepseek_v41` (docker inspect).

Prints one JSON line per request, then a summary line ("passed"); exits 1 when a required
part fails or the endpoint does not answer, 0 otherwise. Stdlib only.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = ("Alice is a 30-year-old structural engineer who lives in Paris. She cycles to work and keeps bees on "
          "her roof. Write a short paragraph introducing her to a new colleague.\n\n")
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
          "required": ["name", "age", "city"], "additionalProperties": False}
MAX_TOKENS = 256


def schema_problems(value, schema: dict, path: str = "$") -> list[str]:
    """The subset of JSON Schema SCHEMA uses: type, properties, required, additionalProperties."""
    kind = schema.get("type")
    checks = {"object": lambda v: isinstance(v, dict), "string": lambda v: isinstance(v, str),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)}
    if kind and not checks[kind](value):
        return [f"{path} is not {kind}"]
    problems = []
    if kind == "object":
        properties = schema.get("properties", {})
        problems += [f"{path}.{key} missing" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"{path}.{key} not allowed" for key in value if key not in properties]
        for key, sub in properties.items():
            if key in value:
                problems += schema_problems(value[key], sub, f"{path}.{key}")
    return problems


def judge(text: str) -> list[str]:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return [f"output does not open with the object: {text[:80]!r}"]
    try:
        value = json.loads(text)
    except ValueError as error:
        return [f"not one JSON value: {error}"]
    return schema_problems(value, SCHEMA)


def post(base: str, body: dict, timeout: float):
    request = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def complete(base: str, body: dict, stream: bool, timeout: float) -> dict:
    body = dict(body, stream=True) if stream else body
    with post(base, body, timeout) as response:
        if not stream:
            choice = json.load(response)["choices"][0]
            return {"text": choice.get("text") or "", "finish_reason": choice.get("finish_reason")}
        text, finish, first = "", None, None
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                piece = choice.get("text") or ""
                if piece and first is None:
                    first = piece
                text += piece
                finish = choice.get("finish_reason") or finish
        return {"text": text, "finish_reason": finish, "first_delta": first}


def reasoning_parser(container: str) -> str | None:
    process = subprocess.run(["docker", "inspect", container], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"docker inspect {container}: {process.stderr.strip()[:200]}")
    info = json.loads(process.stdout)[0]
    argv = [*(info.get("Config", {}).get("Entrypoint") or []), *(info.get("Config", {}).get("Cmd") or []),
            *(info.get("Args") or [])]
    for index, arg in enumerate(argv):
        if arg == "--reasoning-parser" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--reasoning-parser="):
            return arg.split("=", 1)[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--model")
    parser.add_argument("--container", help="the serving container on this host, to check its reasoning parser")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    label = "#1013 completions grammar from the first token"
    summary = {"summary": label, "passed": False}
    problems = []
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            model = args.model or json.load(response)["data"][0]["id"]
    except OSError as error:
        summary["error"] = f"endpoint did not answer: {error}"
        print(json.dumps(summary))
        return 1
    summary["model"] = model
    if args.container:
        try:
            summary["reasoning_parser"] = reasoning_parser(args.container)
        except (RuntimeError, OSError, ValueError) as error:
            summary["reasoning_parser"] = None
            problems.append(f"cannot read the server's reasoning parser: {error}")
        else:
            if summary["reasoning_parser"] != "deepseek_v41":
                problems.append(f"server runs --reasoning-parser {summary['reasoning_parser']}, not deepseek_v41")
    base_body = {"model": model, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0}
    grammars = [("structured_outputs", {"structured_outputs": {"json": SCHEMA}}),
                ("response_format", {"response_format": {"type": "json_schema",
                                                          "json_schema": {"name": "person", "schema": SCHEMA}}})]
    field = None
    rows = {}
    for name, stream in (("grammar", False), ("grammar-stream", True), ("unconstrained", False)):
        started = time.monotonic()
        row = {"request": name}
        try:
            if name == "unconstrained":
                reply = complete(base, base_body, stream, args.timeout)
            else:
                reply = None
                for candidate, extra in grammars if field is None else [g for g in grammars if g[0] == field]:
                    try:
                        reply = complete(base, {**base_body, **extra}, stream, args.timeout)
                    except urllib.error.HTTPError as error:
                        if error.code == 400 and field is None and candidate == "structured_outputs":
                            row.setdefault("refused", []).append(f"{candidate}: {error.read()[:200]!r}")
                            continue
                        raise
                    field = candidate
                    break
                row["grammar_field"] = field
        except (OSError, ValueError, KeyError, IndexError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            reply = None
        row["seconds"] = round(time.monotonic() - started, 3)
        if reply is not None:
            found = judge(reply["text"])
            row.update(text=reply["text"][:400], finish_reason=reply["finish_reason"], schema_valid=not found,
                       problems=found)
            if "first_delta" in reply:
                row["first_delta"] = reply["first_delta"]
            if name != "unconstrained":
                if reply["finish_reason"] != "stop":
                    found = [*found, f"finish_reason {reply['finish_reason']}, not stop"]
                problems += [f"{name}: {p}" for p in found]
        elif name != "unconstrained":
            problems.append(f"{name}: {row['error']}")
        rows[name] = row
        print(json.dumps(row, ensure_ascii=False), flush=True)
    grammar, stream, plain = rows["grammar"], rows["grammar-stream"], rows["unconstrained"]
    summary.update(passed=not problems, problems=problems, grammar_field=field,
                   whole_equals_stream=grammar.get("text") == stream.get("text") if "text" in stream else None,
                   unconstrained_schema_valid=plain.get("schema_valid"),
                   shows_1107_behaviour=bool(grammar.get("schema_valid") and plain.get("schema_valid") is False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
