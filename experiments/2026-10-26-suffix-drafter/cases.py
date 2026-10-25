"""The copy probe's categories (experiments/2026-10-23-next-release/acceptance/copy_probe.py) as
conversations whose assistant reply is the ideal copy: the file reproduced, the file with a
`# reviewed` line above every `def`, an edit_file call whose old_string is the first function.
The 2026-10-23 run kept no token ids; the model's replies there were near-verbatim (a suffix
drafter reached 15.9 of at most 17 tokens a step on the reproduced file), so the ideal reply
stands in for them. Prose has no ideal reply and is left to the measured rates."""

import ast
import json
from pathlib import Path

EDIT_TOOL = {"type": "function", "function": {
    "name": "edit_file", "description": "Replace old_string with new_string in a file.",
    "parameters": {"type": "object", "required": ["path", "old_string", "new_string"], "properties": {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}}
# Prose for the route probe stands in for an agent's own text: this repository's methodology.
PROSE_PROMPT = "Explain how this project measures a serving change before it is promoted."
CATEGORIES = ("copy", "edit", "tool")


def document(prompts, tokens: int, seed: int) -> str:
    """copy_probe.py's document (standard-library source, first line dropped, cut at a line),
    shortened until it parses so that tool has a function to copy."""
    code = prompts.source_text(tokens, seed).split("\n", 1)[1]
    code = code[: code.rfind("\n")]
    while True:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            code = code[: code.rfind("\n")]


def first_function(code: str) -> tuple[str, str]:
    """The first function or method defined in ``code``: its exact source and its name."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines = code.splitlines(keepends=True)
            return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip("\n"), node.name
    raise ValueError("no function in the document")


def conversation(name: str, code: str, repo: Path | None = None, prose_text: str | None = None):
    """``(messages, reply)``: the case's messages before the reply, and the reply."""
    fence = f"```python\n{code}\n```"
    if name == "copy":
        user = f"Here is a Python file:\n\n{fence}\n\nReproduce the file exactly as it is, with no commentary."
        return [{"role": "user", "content": user}], {"role": "assistant", "content": fence}
    if name == "edit":
        user = (f"Here is a Python file:\n\n{fence}\n\nOutput the complete file again, unchanged except for a "
                "line `# reviewed` (at the same indentation) above every line that starts a `def`.")
        out = []
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(("def ", "async def ")):
                out.append(line[: len(line) - len(stripped)] + "# reviewed")
            out.append(line)
        return [{"role": "user", "content": user}], {"role": "assistant", "content": "```python\n" + "\n".join(out) + "\n```"}
    if name == "tool":
        user = (f"The file lib/module.py contains:\n\n{fence}\n\nRename the first function or method defined in "
                "it by adding the suffix `_v2` to its name. Use edit_file; old_string must be that whole "
                "function, copied exactly, and new_string the same function with only its name changed.")
        old, fn = first_function(code)
        new = old.replace(f"def {fn}(", f"def {fn}_v2(", 1)
        call = {"type": "function", "function": {"name": "edit_file", "arguments": json.dumps(
            {"path": "lib/module.py", "old_string": old, "new_string": new})}}
        return ([{"role": "user", "content": user, "tools": [EDIT_TOOL]}],
                {"role": "assistant", "content": "", "tool_calls": [call]})
    if name == "prose":
        return [{"role": "user", "content": PROSE_PROMPT}], {"role": "assistant", "content": prose_text}
    raise ValueError(name)


def render(encode_messages, encode, messages, reply) -> tuple[list[int], int]:
    """Token ids of prompt + reply in the checkpoint's chat format (thinking off), and where the
    reply starts. ``encode`` maps text to ids."""
    prompt_ids = encode(encode_messages(messages, thinking_mode="chat"))
    full_ids = encode(encode_messages(messages + [reply], thinking_mode="chat"))
    assert full_ids[: len(prompt_ids)] == prompt_ids, "the prompt is not a prefix of the conversation"
    return full_ids, len(prompt_ids)
