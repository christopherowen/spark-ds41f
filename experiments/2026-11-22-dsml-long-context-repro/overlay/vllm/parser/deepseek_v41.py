# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls.

DSML has no escaping, so a tool argument that contains a DSML closer (a file *about*
DSML) used to be cut at that closer: the argument was dropped or truncated, text after
it vanished, and the streaming path emitted unterminated JSON.

Rules (structural DSML overlay; kill switch ``VLLM_DSV41_STRUCTURAL_DSML=0``):
  * TOOL_ARGS structural closers: a parameter closer ends a value only when what follows
    is whitespace and then a new parameter head, or the end of the call's arguments.
    An invoke/calls closer ends a call only when its arguments are closed; otherwise it
    is held until the next tokens prove whether structure follows.
  * Token provenance is kept unchanged on auto/unconstrained tool choice: markup spelled
    in ordinary pieces stays text, and only markup carrying a genuine ``｜DSML｜`` token
    (128825) is a candidate structural boundary. Under forced/named ``tool_choice``,
    BPE-spelled DSML inside ``<think>`` stays reasoning prose while post-``</think>``
    constrained DSML markup is accepted even if spelled with ordinary tokens.
  * Dialect/spacing variants carrying genuine ``｜DSML｜`` (``<｜DSML｜calls>``,
    ``<｜DSML｜ call>``, ``<｜DSML｜ tool_calls>``, V4 unspaced ``<｜DSML｜tool_calls>`` /
    ``<｜DSML｜invoke name="`` / ``<｜DSML｜parameter``, ``function_calls`` / ``function``,
    optional whitespace before ``>``, and closers) are accepted alongside canonical V4.1.
  * ``TOOL_PREAMBLE`` exits cleanly on ``TOOL_END`` or non-whitespace plain text, and an
    orphan ``INVOKE_PREFIX`` inside ``REASONING`` implicitly closes reasoning.
  * Genuine-``｜DSML｜`` headless ``<｜DSML｜ parameter ...>`` blocks emitted outside
    ``TOOL_ARGS`` recover into tool calls when declared ``request.tools`` schemas
    uniquely identify the target tool (or when the first parameter substitutes
    ``parameter`` for ``invoke``), and are quarantined (dropped from ``content``
    without guessing) when ambiguous or undeclared.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace

import regex as re
from openai.types.responses import CustomTool, FunctionTool, NamespaceTool

from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _dsml_arg_value_extender,
    _dsml_open_value_extender,
    _param_patterns,
    _unwrap_wrapper_args,
    deepseek_v4_config,
)
from vllm.parser.engine.adapters import (
    ParserEngineReasoningAdapter,
    ParserEngineToolAdapter,
)
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import _TokenOriginEvent
from vllm.parser.engine.token_id_scanner import DROP_TERMINAL
from vllm.parser.metrics import record_tool_parser_invocation
from vllm.tool_parsers.utils import (
    _extract_tool_info,
    _is_function_tool,
    iter_response_function_tool_info,
)

logger = init_logger(__name__)

DSML_ATOM = "｜DSML｜"

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

DSML_TOOL_START_VARIANTS: tuple[str, ...] = (
    DSML_TOOL_START,
    "<｜DSML｜calls>",
    "<｜DSML｜ call>",
    "<｜DSML｜call>",
    "<｜DSML｜ tool_calls>",
    "<｜DSML｜tool_calls>",
    "<｜DSML｜ toolcalls>",
    "<｜DSML｜toolcalls>",
    "<｜DSML｜ tool>",
    "<｜DSML｜tool>",
    "<｜DSML｜ function_calls>",
    "<｜DSML｜function_calls>",
)
DSML_TOOL_END_VARIANTS: tuple[str, ...] = (
    DSML_TOOL_END,
    "</｜DSML｜calls>",
    "</｜DSML｜ call>",
    "</｜DSML｜call>",
    "</｜DSML｜ tool_calls>",
    "</｜DSML｜tool_calls>",
    "</｜DSML｜ toolcalls>",
    "</｜DSML｜toolcalls>",
    "</｜DSML｜ tool>",
    "</｜DSML｜tool>",
    "</｜DSML｜ function_calls>",
    "</｜DSML｜function_calls>",
)
DSML_INVOKE_PREFIX_VARIANTS: tuple[str, ...] = (
    DSML_INVOKE_PREFIX,
    '<｜DSML｜invoke name="',
    '<｜DSML｜ function name="',
    '<｜DSML｜function name="',
)
DSML_INVOKE_NAME_END_VARIANTS: tuple[str, ...] = (
    DSML_INVOKE_NAME_END,
    '" >',
)
DSML_INVOKE_END_VARIANTS: tuple[str, ...] = (
    DSML_INVOKE_END,
    "</｜DSML｜invoke>",
    "</｜DSML｜ function>",
    "</｜DSML｜function>",
)
DSML_PARAM_START_VARIANTS: tuple[str, ...] = (
    DSML_PARAM_START,
    "<｜DSML｜parameter",
)
DSML_PARAM_CLOSE_VARIANTS: tuple[str, ...] = (
    DSML_PARAM_CLOSE,
    "</｜DSML｜parameter>",
)

_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)
_LEGACY_PARAM_HEAD_RE = re.compile(
    re.escape(DSML_PARAM_START) + r'\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
)
_LEGACY_PARAM_CLOSE_RE = re.compile(re.escape(DSML_PARAM_CLOSE))

_PARAM_START_PAT = rf"<{re.escape(DSML_ATOM)} ?parameter"
_PARAM_CLOSE_PAT = rf"</{re.escape(DSML_ATOM)} ?parameter>"
_PARAM_CLOSE_RE = re.compile(_PARAM_CLOSE_PAT)
_PARAM_HEAD_RE = re.compile(
    rf'{_PARAM_START_PAT}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
)

OVERLAY_ID = "dsv41-structural-dsml-v1"

# ── structural closers ──────────────────────────────────────

_NEXT_HEAD_RE = re.compile(rf'{_PARAM_START_PAT}\s+name="')
_HEAD_IN_PROGRESS_RE = re.compile(rf"{_PARAM_START_PAT}(?:\s[^>]*)?$")
_PARAM_CLOSE_ATOM = DSML_PARAM_CLOSE.find(DSML_ATOM)


def _closer_is_structural(raw: str, after: int, partial: bool) -> bool:
    follower = raw[after:].lstrip()
    if not follower:
        return True
    if _NEXT_HEAD_RE.match(follower):
        return True
    if partial and (
        any(v.startswith(follower) for v in DSML_PARAM_START_VARIANTS)
        or _HEAD_IN_PROGRESS_RE.match(follower)
    ):
        return True
    return False


def _find_value_end(raw: str, vstart: int, partial: bool) -> tuple[int, int] | None:
    """Return (value_end, resume_pos) for the value starting at vstart, or None
    when the value is still open (no structural closer and no next head)."""
    pos = vstart
    head = _NEXT_HEAD_RE.search(raw, vstart)
    while True:
        m_close = _PARAM_CLOSE_RE.search(raw, pos)
        c = m_close.start() if m_close is not None else -1
        if head is not None and (c < 0 or head.start() < c):
            return head.start(), head.start()
        if m_close is None:
            return None
        after = m_close.end()
        if _closer_is_structural(raw, after, partial):
            return c, after
        pos = after


def _open_value_end(raw: str, vstart: int) -> int:
    """End of the open value that starts at vstart."""
    search_from = max(vstart, raw.rfind(">") + 1)
    cuts = [raw.find(v, search_from) for v in DSML_PARAM_START_VARIANTS]
    valid = [c for c in cuts if c >= 0]
    return min(valid) if valid else len(raw)


def _v41_open_value_extender(raw_args: str) -> ArgValueExtender:
    keep = max(len(DSML_PARAM_START), len(DSML_PARAM_CLOSE)) - 1
    tail = raw_args[-keep:]
    stop_tags = (*DSML_PARAM_START_VARIANTS, *DSML_PARAM_CLOSE_VARIANTS)

    def extend(delta: str) -> str | None:
        nonlocal tail
        text = tail + delta
        if any(tag in text for tag in stop_tags):
            return None
        tail = text[-keep:]
        return json.dumps(delta, ensure_ascii=False)[1:-1]

    return extend


def _store(params: dict, name: str, is_str: str | None, value: str, partial: bool):
    if is_str == "true":
        params[name] = value
        return
    try:
        params[name] = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        if not partial:
            params[name] = value


def _structural_dsml_arg_converter(raw_args: str, partial: bool) -> str:
    """Stock's text-path converter, with the structural rule for closers."""
    params: dict[str, object] = {}
    pos = 0
    while True:
        m = _PARAM_HEAD_RE.search(raw_args, pos)
        if not m:
            break
        name, is_str = m.group(1), m.group(2)
        found = _find_value_end(raw_args, m.end(), partial)
        if found is None:
            if partial:
                value = raw_args[m.end() : _open_value_end(raw_args, m.end())]
                if is_str == "true":
                    params[name] = value
                else:
                    try:
                        params[name] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
            break
        vend, pos = found
        _store(params, name, is_str, raw_args[m.end() : vend], partial=False)
    return json.dumps(params, ensure_ascii=False)


def _structural_scan(raw_args: str) -> tuple[set[str], int]:
    """Names of structurally complete parameters, and the end of the last one."""
    names: set[str] = set()
    last_end = 0
    pos = 0
    while True:
        m = _PARAM_HEAD_RE.search(raw_args, pos)
        if not m:
            return names, last_end
        found = _find_value_end(raw_args, m.end(), partial=False)
        if found is None:
            return names, last_end
        _vend, pos = found
        names.add(m.group(1))
        last_end = pos


def _structural_open_value_extender(raw_args: str) -> ArgValueExtender | None:
    """Stock's text-path extender rule, decided with the structural rule."""
    names, last_end = _structural_scan(raw_args)
    head = _PARAM_HEAD_RE.search(raw_args, last_end)
    if head is None or _find_value_end(raw_args, head.end(), partial=False) is not None:
        return None
    if head.group(2) != "true" or head.group(1) in names:
        return None
    if _open_value_end(raw_args, head.end()) < len(raw_args):
        return None
    return _v41_open_value_extender(raw_args)


# ── token provenance (candidate-60) ────────────────────────────────────────


def _token_origin_boundaries(
    raw_args: str,
    spans: Iterable[tuple[int, int]],
    *,
    allow_variants: bool = False,
) -> list[tuple[int, int, re.Match | None]]:
    """Parameter tags whose DSML atom came from the DSML token, in text order."""
    trusted = set(spans)
    head_re = _PARAM_HEAD_RE if allow_variants else _LEGACY_PARAM_HEAD_RE
    close_re = _PARAM_CLOSE_RE if allow_variants else _LEGACY_PARAM_CLOSE_RE
    boundaries = []
    for head in head_re.finditer(raw_args, partial=True):
        atom_start = head.start() + 1
        if (atom_start, atom_start + len(DSML_ATOM)) in trusted:
            boundaries.append(
                (head.start(), head.end(), None if head.partial else head)
            )
    for close in close_re.finditer(raw_args):
        atom_start = close.start() + 2
        if (atom_start, atom_start + len(DSML_ATOM)) in trusted:
            boundaries.append((close.start(), close.end(), None))
    boundaries.sort(key=lambda entry: entry[0])
    return boundaries


def _text_structural_boundaries(
    raw_args: str,
    partial: bool,
) -> list[tuple[int, int, re.Match | None]]:
    """Text-mode structural boundaries matching ``_structural_token_origin_boundaries``."""
    boundaries: list[tuple[int, int, re.Match | None]] = []
    for head in _PARAM_HEAD_RE.finditer(raw_args, partial=True):
        boundaries.append((head.start(), head.end(), None if head.partial else head))
    for close in _PARAM_CLOSE_RE.finditer(raw_args):
        if _closer_is_structural(raw_args, close.end(), partial):
            boundaries.append((close.start(), close.end(), None))
    boundaries.sort(key=lambda entry: entry[0])
    return boundaries


def _args_from_boundaries(
    raw_args: str,
    partial: bool,
    boundaries: Sequence[tuple[int, int, re.Match | None]],
) -> str:
    """Parse raw arguments from a boundary list (candidate-60's body)."""
    params = {}
    for index, (_, end, head) in enumerate(boundaries):
        if head is None:
            continue
        complete = index + 1 < len(boundaries)
        if not complete and not partial:
            continue
        value_end = boundaries[index + 1][0] if complete else len(raw_args)
        value = raw_args[end:value_end]
        name, is_str = head.group(1), head.group(2)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                if complete:
                    params[name] = value
    return json.dumps(params, ensure_ascii=False)


def _token_origin_args(
    raw_args: str, partial: bool, spans: Iterable[tuple[int, int]]
) -> str:
    """Parse only parameter boundaries carrying a genuine DSML token atom."""
    return _args_from_boundaries(
        raw_args, partial, _token_origin_boundaries(raw_args, spans)
    )


def _structural_token_origin_boundaries(
    raw_args: str, spans: Iterable[tuple[int, int]], partial: bool
) -> list[tuple[int, int, re.Match | None]]:
    """Token-origin boundaries, keeping a closer only when it is structural."""
    return [
        entry
        for entry in _token_origin_boundaries(raw_args, spans, allow_variants=True)
        if entry[2] is not None or _closer_is_structural(raw_args, entry[1], partial)
    ]


def _open_extender_for_boundaries(
    raw_args: str,
    boundaries: Sequence[tuple[int, int, re.Match | None]],
    *,
    allow_variants: bool = False,
):
    """candidate-60's open-value test over a boundary list."""
    if not boundaries or (head := boundaries[-1][2]) is None:
        return None
    names = {entry[2].group(1) for entry in boundaries[:-1] if entry[2]}
    if head.group(2) != "true" or head.group(1) in names:
        return None
    if allow_variants:
        return _v41_open_value_extender(raw_args)
    return _dsml_open_value_extender(raw_args, DSML_PARAM_START, DSML_PARAM_CLOSE)


def _iter_declared_tool_schemas(
    tools,
) -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """Return ``(name, properties, required)`` for each declared tool."""
    if not tools:
        return []
    out: list[tuple[str, frozenset[str], frozenset[str]]] = []
    for tool in tools:
        if isinstance(tool, (FunctionTool, NamespaceTool, CustomTool)):
            for name, params in iter_response_function_tool_info(tool):
                p = params or {}
                out.append(
                    (
                        name,
                        frozenset((p.get("properties") or {}).keys()),
                        frozenset(p.get("required") or ()),
                    )
                )
        elif _is_function_tool(tool):
            name, params = _extract_tool_info(tool)
            p = params or {}
            out.append(
                (
                    name,
                    frozenset((p.get("properties") or {}).keys()),
                    frozenset(p.get("required") or ()),
                )
            )
        elif isinstance(tool, dict):
            fn = (
                tool.get("function")
                if isinstance(tool.get("function"), dict)
                else tool
            )
            name = fn.get("name")
            if isinstance(name, str) and name:
                p = fn.get("parameters") or {}
                out.append(
                    (
                        name,
                        frozenset((p.get("properties") or {}).keys()),
                        frozenset(p.get("required") or ()),
                    )
                )
    return out


def _resolve_headless_dsml_call(
    raw_args: str,
    spans: Sequence[tuple[int, int]],
    *,
    token_origin: bool,
    tools,
) -> tuple[str, str, tuple[tuple[int, int], ...]] | None:
    """Resolve a headless ``<｜DSML｜ parameter ...>`` block to a unique tool."""
    boundaries = (
        _structural_token_origin_boundaries(raw_args, spans, partial=False)
        if token_origin
        else _text_structural_boundaries(raw_args, partial=False)
    )
    param_heads = [
        (idx, end, head)
        for idx, (_, end, head) in enumerate(boundaries)
        if head is not None
    ]
    schemas = _iter_declared_tool_schemas(tools)
    if not param_heads or not schemas:
        return None

    keys = frozenset(h.group(1) for _, _, h in param_heads)
    key_matches = [
        t_name for t_name, props, req in schemas if props and req <= keys <= props
    ]
    if len(key_matches) == 1:
        return key_matches[0], raw_args, tuple(spans)
    if len(key_matches) > 1:
        return None

    first_idx, first_end, first_head = param_heads[0]
    first_name = first_head.group(1)
    next_boundary_start = (
        boundaries[first_idx + 1][0]
        if first_idx + 1 < len(boundaries)
        else len(raw_args)
    )
    first_val = raw_args[first_end:next_boundary_start]
    if not first_val.strip():
        rest_keys = frozenset(h.group(1) for _, _, h in param_heads[1:])
        sub_matches = [
            t_name
            for t_name, props, req in schemas
            if t_name == first_name
            and (len(param_heads) > 1 or not props)
            and req <= rest_keys
            and (not props or rest_keys <= props)
        ]
        if len(sub_matches) == 1:
            body_raw = raw_args[next_boundary_start:]
            body_spans = tuple(
                (s - next_boundary_start, e - next_boundary_start)
                for s, e in spans
                if s >= next_boundary_start
            )
            return sub_matches[0], body_raw, body_spans

    return None


# ── engine guard ───────────────────────────────────────────────────────────

_ENABLED = os.environ.get("VLLM_DSV41_STRUCTURAL_DSML", "1") != "0"
_V41Engine = None

try:
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

    _REQUIRED = (
        "_on_terminal",
        "_emit_for_state",
        "finish",
        "reset",
        "_process_scanner_items",
        "_process_lex_tokens",
        "_record_reasoning_tokens",
    )
    _missing = [a for a in _REQUIRED if not hasattr(StreamingParserEngine, a)]
    if _missing:
        raise AttributeError(f"StreamingParserEngine lacks {_missing}")
    for _hook in ("_on_terminal", "_emit_for_state"):
        if "token_id_spans" not in inspect.signature(
            getattr(StreamingParserEngine, _hook)
        ).parameters:
            raise AttributeError(f"StreamingParserEngine.{_hook} lacks token_id_spans")

    _HEADLESS_ENTRY_STATES = frozenset(
        {
            ParserState.CONTENT,
            ParserState.REASONING,
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_BETWEEN,
        }
    )

    class _V41StructuralEngine(StreamingParserEngine):
        """Structural DSML closer lookahead, headless recovery, and forced-choice guard."""

        _v41_args = ""
        _v41_held = None
        _v41_held_ws = ""
        _v41_headless_active = False
        _v41_headless_raw = ""
        _v41_headless_spans: list[tuple[int, int]] = []
        _v41_headless_held = None
        _v41_headless_held_ws = ""
        _v41_headless_resolver = None
        _v41_headless_suppressed = None
        _v41_forced_tool_choice = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            orig_accept = self._lexer._accept_match

            def _accept_match(terminal: str, literal: str) -> bool:
                if (
                    self._v41_forced_tool_choice
                    and self.state != ParserState.REASONING
                ):
                    return True
                return orig_accept(terminal, literal)

            self._lexer._accept_match = _accept_match

        def reset(self, initial_state=None) -> None:
            super().reset(initial_state)
            self._v41_args = ""
            self._v41_held = None
            self._v41_held_ws = ""
            self._v41_headless_active = False
            self._v41_headless_raw = ""
            self._v41_headless_spans = []
            self._v41_headless_held = None
            self._v41_headless_held_ws = ""

        def _v41_token_origin(self) -> bool:
            if self._v41_forced_tool_choice:
                return False
            return bool(self._ever_had_token_ids and self._has_composite_ids)

        @staticmethod
        def _v41_raw_closed(raw: str) -> bool:
            s = raw.rstrip()
            return (
                (not s)
                or not any(v in s for v in DSML_PARAM_START_VARIANTS)
                or any(s.endswith(v) for v in DSML_PARAM_CLOSE_VARIANTS)
            )

        @staticmethod
        def _v41_raw_open_param(raw: str) -> bool:
            i = max(raw.rfind(v) for v in DSML_PARAM_START_VARIANTS)
            if i < 0:
                return False
            m = _PARAM_HEAD_RE.match(raw, i)
            if not m:
                return False
            return _find_value_end(raw, m.end(), partial=False) is None

        def _v41_args_closed(self) -> bool:
            return self._v41_raw_closed(self._v41_args)

        def _v41_args_open_param(self) -> bool:
            return self._v41_raw_open_param(self._v41_args)

        def _v41_is_headless_suppressed(self) -> bool:
            return bool(
                self.skip_tool_parsing
                or self._v41_headless_resolver is None
                or (
                    self._v41_headless_suppressed is not None
                    and self._v41_headless_suppressed()
                )
            )

        def _v41_append_headless(self, text: str, token_id_spans=()) -> None:
            if not text:
                return
            base = len(self._v41_headless_raw)
            if token_id_spans:
                self._v41_headless_spans.extend(
                    (start + base, end + base) for start, end in token_id_spans
                )
            self._v41_headless_raw += text

        def _v41_start_headless(self, value: str, token_id_spans=()) -> None:
            self._v41_headless_active = True
            self._v41_headless_raw = ""
            self._v41_headless_spans = []
            self._v41_headless_held = None
            self._v41_headless_held_ws = ""
            self._v41_append_headless(value, token_id_spans)

        def _v41_release_headless_literal(self) -> None:
            _, value, _, token_id_spans = self._v41_headless_held
            ws = self._v41_headless_held_ws
            self._v41_headless_held = None
            self._v41_headless_held_ws = ""
            self._v41_append_headless(value, token_id_spans)
            if ws:
                self._v41_append_headless(ws)

        def _v41_close_open_headless_param(self) -> None:
            if self._v41_raw_open_param(self._v41_headless_raw):
                spans = (
                    ((_PARAM_CLOSE_ATOM, _PARAM_CLOSE_ATOM + len(DSML_ATOM)),)
                    if self._v41_token_origin()
                    else ()
                )
                self._v41_append_headless(DSML_PARAM_CLOSE, spans)

        def _v41_flush_headless(self) -> list[SemanticEvent]:
            raw = self._v41_headless_raw
            spans = tuple(self._v41_headless_spans)
            self._v41_headless_active = False
            self._v41_headless_raw = ""
            self._v41_headless_spans = []
            self._v41_headless_held = None
            self._v41_headless_held_ws = ""

            self._tool_between_ws = ""
            self._tool_between_has_nonws = False
            self.state = ParserState.TOOL_BETWEEN

            resolved = (
                self._v41_headless_resolver(raw, spans)
                if self._v41_headless_resolver is not None
                else None
            )
            if resolved is None:
                logger.warning(
                    "%s: quarantined unresolved headless DSML parameter block (%d chars)",
                    OVERLAY_ID,
                    len(raw),
                )
                return []

            tool_name, body_raw, body_spans = resolved
            logger.info(
                "%s: recovered headless DSML parameter block as tool %r",
                OVERLAY_ID,
                tool_name,
            )
            self.tool_index += 1
            events: list[SemanticEvent] = [
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=self.tool_index),
                SemanticEvent(
                    EventType.TOOL_NAME,
                    value=tool_name,
                    tool_index=self.tool_index,
                ),
            ]
            if body_raw:
                ev_cls = _TokenOriginEvent if body_spans else SemanticEvent
                origin = {"token_id_spans": body_spans} if body_spans else {}
                events.append(
                    ev_cls(
                        EventType.ARG_VALUE_CHUNK,
                        value=body_raw,
                        tool_index=self.tool_index,
                        **origin,
                    )
                )
            events.append(
                SemanticEvent(EventType.TOOL_CALL_END, tool_index=self.tool_index)
            )
            return events

        def _emit_for_state(self, text: str, token_count: int = 0, token_id_spans=()):
            if self._v41_headless_active:
                if self._v41_headless_held is not None:
                    if not text.strip():
                        self._v41_headless_held_ws += text
                        return []
                    self._v41_release_headless_literal()
                self._v41_append_headless(text, token_id_spans)
                return []
            if self.state != ParserState.TOOL_ARGS:
                return super()._emit_for_state(text, token_count, token_id_spans)
            events = []
            if self._v41_held is not None:
                if not text.strip():
                    self._v41_held_ws += text
                    return events
                events.extend(self._v41_release_literal())
            self._v41_args += text
            events.extend(
                super()._emit_for_state(text, token_count, token_id_spans)
            )
            return events

        def _v41_release_literal(self):
            terminal, value, _, _ = self._v41_held
            text = value + self._v41_held_ws
            self._v41_held = None
            self._v41_held_ws = ""
            logger.info(
                "%s: DSML %s inside a parameter value kept as text (call %d)",
                OVERLAY_ID,
                terminal,
                self.tool_index,
            )
            self._v41_args += text
            return StreamingParserEngine._emit_for_state(self, text, 0)

        def _v41_resolve_structural(self):
            terminal, value, token_count, token_id_spans = self._v41_held
            ws = self._v41_held_ws
            self._v41_held = None
            self._v41_held_ws = ""
            events = []
            if self._v41_args_open_param():
                spans = (
                    ((_PARAM_CLOSE_ATOM, _PARAM_CLOSE_ATOM + len(DSML_ATOM)),)
                    if self._v41_token_origin()
                    else ()
                )
                self._v41_args += DSML_PARAM_CLOSE
                events.extend(
                    StreamingParserEngine._emit_for_state(
                        self, DSML_PARAM_CLOSE, 0, spans
                    )
                )
            events.extend(
                super()._on_terminal(terminal, value, token_count, token_id_spans)
            )
            if ws:
                events.extend(self._emit_for_state(ws, 0))
            return events

        def _on_terminal(
            self, terminal: str, value: str, token_count: int = 0, token_id_spans=()
        ):
            if self._v41_headless_active:
                events: list[SemanticEvent] = []
                if self._v41_headless_held is not None:
                    if terminal in ("TOOL_END", "INVOKE_PREFIX", "TOOL_START"):
                        ws = self._v41_headless_held_ws
                        self._v41_headless_held = None
                        self._v41_headless_held_ws = ""
                        self._v41_close_open_headless_param()
                        events.extend(self._v41_flush_headless())
                        if ws:
                            events.extend(self._emit_for_state(ws, 0))
                        events.extend(
                            self._on_terminal(
                                terminal, value, token_count, token_id_spans
                            )
                        )
                        return events
                    self._v41_release_headless_literal()

                if terminal in ("PARAM_START", "PARAM_CLOSE", "INVOKE_NAME_END"):
                    self._v41_append_headless(value, token_id_spans)
                    return events
                if terminal in ("INVOKE_END", "TOOL_END"):
                    if not self._v41_raw_closed(self._v41_headless_raw):
                        self._v41_headless_held = (
                            terminal,
                            value,
                            token_count,
                            token_id_spans,
                        )
                        self._v41_headless_held_ws = ""
                        return events
                    events.extend(self._v41_flush_headless())
                    return events
                if terminal in ("INVOKE_PREFIX", "TOOL_START"):
                    self._v41_close_open_headless_param()
                    events.extend(self._v41_flush_headless())
                    events.extend(
                        self._on_terminal(
                            terminal, value, token_count, token_id_spans
                        )
                    )
                    return events
                if self._has_drops and terminal == DROP_TERMINAL:
                    return events
                self._v41_append_headless(value, token_id_spans)
                return events

            if (
                terminal == "PARAM_START"
                and self.state in _HEADLESS_ENTRY_STATES
                and not self._v41_is_headless_suppressed()
            ):
                events = []
                if self.state == ParserState.REASONING:
                    events.append(
                        SemanticEvent(
                            EventType.REASONING_END, tool_index=self.tool_index
                        )
                    )
                self._v41_start_headless(value, token_id_spans)
                return events

            if self.state == ParserState.TOOL_ARGS:
                events = []
                if self._v41_held is not None:
                    if terminal in ("TOOL_END", "INVOKE_PREFIX"):
                        events.extend(self._v41_resolve_structural())
                        events.extend(
                            self._on_terminal(
                                terminal, value, token_count, token_id_spans
                            )
                        )
                        return events
                    events.extend(self._v41_release_literal())
                if terminal in (
                    "INVOKE_END",
                    "TOOL_END",
                ) and not self._v41_args_closed():
                    self._v41_held = (terminal, value, token_count, token_id_spans)
                    self._v41_held_ws = ""
                    return events
                events.extend(
                    self._v41_on_terminal_tracked(
                        terminal, value, token_count, token_id_spans
                    )
                )
                return events
            return self._v41_on_terminal_tracked(
                terminal, value, token_count, token_id_spans
            )

        def _v41_on_terminal_tracked(
            self, terminal, value, token_count, token_id_spans
        ):
            before = self.state
            events = super()._on_terminal(
                terminal, value, token_count, token_id_spans
            )
            if before != ParserState.TOOL_ARGS and self.state == ParserState.TOOL_ARGS:
                self._v41_args = ""
                self._v41_held = None
                self._v41_held_ws = ""
            return events

        def finish(self):
            if self._v41_held is None and not self._v41_headless_active:
                return super().finish()
            events = self._process_scanner_items(self._scanner.flush_pending())
            events.extend(self._process_lex_tokens(self._lexer.flush()))
            if self._v41_headless_active:
                ws = self._v41_headless_held_ws
                self._v41_headless_held = None
                self._v41_headless_held_ws = ""
                self._v41_close_open_headless_param()
                events.extend(self._v41_flush_headless())
                if ws:
                    events.extend(self._emit_for_state(ws, 0))
            if self._v41_held is not None and self.state == ParserState.TOOL_ARGS:
                events.extend(self._v41_resolve_structural())
            self._record_reasoning_tokens(events)
            events.extend(super().finish())
            return events

    _V41Engine = _V41StructuralEngine
except Exception as e:  # noqa: BLE001
    logger.error(
        "%s: structural DSML overlay disabled (stock behaviour): %s", OVERLAY_ID, e
    )
    _V41Engine = None

_STRUCTURAL_ACTIVE = _ENABLED and _V41Engine is not None


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    canonical_overrides = {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    if _STRUCTURAL_ACTIVE:
        terminal_overrides: dict[str, str | tuple[str, ...]] = {
            "TOOL_START": DSML_TOOL_START_VARIANTS,
            "TOOL_END": DSML_TOOL_END_VARIANTS,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX_VARIANTS,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END_VARIANTS,
            "INVOKE_END": DSML_INVOKE_END_VARIANTS,
            "PARAM_START": DSML_PARAM_START_VARIANTS,
            "PARAM_CLOSE": DSML_PARAM_CLOSE_VARIANTS,
        }
        transitions = {
            **config.transitions,
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        non_ws_transitions = {
            **config.non_whitespace_transitions,
            ParserState.TOOL_PREAMBLE: Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            ),
        }
        converter = _structural_dsml_arg_converter
    else:
        terminal_overrides = dict(canonical_overrides)
        transitions = config.transitions
        non_ws_transitions = config.non_whitespace_transitions
        converter = functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
            param_start=DSML_PARAM_START,
        )
    return replace(
        config,
        name="deepseek_v41",
        turn_boundary_tokens=frozenset(
            {"<｜Assistant｜>", "<｜User｜>", "<｜System｜>"}
        ),
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            **{
                key: canonical_overrides.get(key, value)
                for key, value in config.token_id_terminals.items()
            },
            "DSML_ATOM": DSML_ATOM,
        },
        composite_token_id_terminals=dict.fromkeys(canonical_overrides, "DSML_ATOM"),
        transitions=transitions,
        non_whitespace_transitions=non_ws_transitions,
        arg_converter=converter,
        arg_value_extender=functools.partial(
            _dsml_arg_value_extender,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
            param_start=DSML_PARAM_START,
            param_close=DSML_PARAM_CLOSE,
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        self._arg_value_extender = self._extend_open_value
        self._v41_forced_tool_choice = False
        if _STRUCTURAL_ACTIVE:
            self._engine = _V41Engine(
                self.parser_engine_config, tokenizer, vocab=self.vocab
            )
            self._engine._v41_headless_resolver = self._resolve_headless_call
            self._engine._v41_headless_suppressed = lambda: bool(
                self._suppress_tool_calls
            )

    def _check_skip_tool_parsing(self, request) -> None:
        super()._check_skip_tool_parsing(request)
        tool_choice = getattr(request, "tool_choice", None)
        forced = bool(
            _STRUCTURAL_ACTIVE
            and not self.skip_tool_parsing
            and not self._suppress_tool_calls
            and tool_choice not in (None, "auto", "none")
        )
        self._v41_forced_tool_choice = forced
        if _STRUCTURAL_ACTIVE:
            self._engine._v41_forced_tool_choice = forced

    @property
    def _token_origin(self) -> bool:
        if self._v41_forced_tool_choice:
            return False
        return self._engine._ever_had_token_ids and self._engine._has_composite_ids

    def _resolve_headless_call(
        self, raw_args: str, spans: Sequence[tuple[int, int]]
    ) -> tuple[str, str, tuple[tuple[int, int], ...]] | None:
        return _resolve_headless_dsml_call(
            raw_args,
            spans,
            token_origin=self._token_origin,
            tools=self._tools,
        )

    def _arg_slot(self, raw_args: str):
        return next(
            (slot for slot in reversed(self._tool_slots) if slot.args is raw_args), None
        )

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        if not self._token_origin:
            return super()._convert_args(raw_args, partial)
        slot = self._arg_slot(raw_args)
        spans = slot.arg_token_id_spans if slot is not None else ()
        if _STRUCTURAL_ACTIVE:
            result = _args_from_boundaries(
                raw_args,
                partial,
                _structural_token_origin_boundaries(raw_args, spans or (), partial),
            )
        else:
            result = _token_origin_args(raw_args, partial, spans or ())
        return _unwrap_wrapper_args(
            result, self._tools, slot.name if slot is not None else None
        )

    def _extend_open_value(self, raw_args: str):
        extender = self.parser_engine_config.arg_value_extender
        assert extender is not None
        if _STRUCTURAL_ACTIVE:
            return self._structural_extend_open_value(raw_args)
        if not self._token_origin:
            return extender(raw_args)
        slot = self._arg_slot(raw_args)
        spans = slot.arg_token_id_spans if slot is not None else ()
        boundaries = _token_origin_boundaries(raw_args, spans or ())
        return _open_extender_for_boundaries(raw_args, boundaries)

    def _structural_extend_open_value(self, raw_args: str):
        if not self._token_origin:
            return _structural_open_value_extender(raw_args)
        slot = self._arg_slot(raw_args)
        spans = slot.arg_token_id_spans if slot is not None else ()
        boundaries = _structural_token_origin_boundaries(raw_args, spans or (), True)
        return _open_extender_for_boundaries(
            raw_args, boundaries, allow_variants=True
        )

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self.set_prompt_token_ids(prompt_token_ids)
        self._reset()


class DeepSeekV41ServingParser(DelegatingParser):
    """Keep serving adapters and request constraints around one output engine."""

    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, **kwargs)
        reasoning_adapter = self._reasoning_parser
        tool_adapter = self._tool_parser
        assert reasoning_adapter is None or isinstance(
            reasoning_adapter, ParserEngineReasoningAdapter
        )
        assert tool_adapter is None or isinstance(
            tool_adapter, ParserEngineToolAdapter
        )
        assert reasoning_adapter is not None or tool_adapter is not None
        self._reasoning_adapter = reasoning_adapter
        if reasoning_adapter is not None:
            self._unified_parser = reasoning_adapter._parser_engine
            self._unified_parser._tools = tools
            if tool_adapter is not None:
                tool_adapter._parser_engine = self._unified_parser
            else:
                self._unified_parser.skip_tool_parsing = True
        else:
            assert tool_adapter is not None
            self._unified_parser = tool_adapter._parser_engine
        self._unified_parser._stream_state = self._stream_state
        self._skip_tools_override = self._tool_parser is None

    @property
    def skip_tool_parsing(self):
        return self._unified_parser.skip_tool_parsing

    @skip_tool_parsing.setter
    def skip_tool_parsing(self, value):
        self._skip_tools_override = value
        self._unified_parser.skip_tool_parsing = value

    def _prepare_tools(self, request, enable_auto_tools=True):
        self._unified_parser.skip_tool_parsing = (
            self._skip_tools_override
            or self._tool_parser is None
            or request.tool_choice == "none"
            or not request.tools
            or not enable_auto_tools
        )
        self._unified_parser._suppress_tool_calls = False

    def _sync_reasoning(self):
        if self._reasoning_adapter is not None:
            self._reasoning_adapter._streaming_count_valid = True
            self._stream_state.reasoning_ended = self._unified_parser.reasoning_ended
        else:
            self._stream_state.reasoning_ended = True

    def set_prompt_token_ids(self, prompt_token_ids):
        self._unified_parser.set_prompt_token_ids(prompt_token_ids)
        if self._reasoning_adapter is not None:
            self._reasoning_adapter._streaming_count_valid = False

    def _uses_json_tool_choice(self, request):
        return (
            self._tool_parser is not None
            and self._tool_parser.supports_required_and_named
            and request.tool_choice not in (None, "auto", "none")
        )

    def parse(
        self, model_output, request, enable_auto_tools=False, model_output_token_ids=()
    ):
        if self._uses_json_tool_choice(request):
            if self._reasoning_adapter is not None:
                with self._reasoning_adapter._skip_tool_parsing():
                    reasoning, content, _ = self._unified_parser.parse(
                        model_output, request, enable_auto_tools, model_output_token_ids
                    )
                self._sync_reasoning()
            else:
                reasoning, content = None, model_output
            tool_calls, content = self._extract_tool_calls(
                content, request, enable_auto_tools
            )
            return reasoning, content, tool_calls
        self._prepare_tools(request, enable_auto_tools)
        is_tool_called: bool | Exception = False
        try:
            result = self._unified_parser.parse(
                model_output, request, enable_auto_tools, model_output_token_ids
            )
            is_tool_called = bool(result[2])
            self._sync_reasoning()
            return result
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            if self._tool_parser is not None:
                record_tool_parser_invocation(
                    is_tool_called=is_tool_called, is_streaming=False, request=request
                )

    def parse_delta(
        self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished
    ):
        if self._uses_json_tool_choice(request):
            self._stream_state.engine_based = False
            return super().parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
        self._prepare_tools(request)
        is_tool_called: bool | Exception = False
        try:
            result = self._unified_parser.parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
            is_tool_called = bool(result and result.tool_calls)
            self._sync_reasoning()
            return result
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            if self._tool_parser is not None:
                record_tool_parser_invocation(
                    is_tool_called=is_tool_called, is_streaming=True, request=request
                )


if _STRUCTURAL_ACTIVE:
    logger.info("%s: structural DSML closers active", OVERLAY_ID)
