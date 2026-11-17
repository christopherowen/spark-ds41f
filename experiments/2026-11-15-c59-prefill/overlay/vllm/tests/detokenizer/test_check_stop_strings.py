# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for check_stop_strings.

These are pure-function tests (no model / GPU). They pin down which stop
string is selected when several stop strings match within the text that was
appended in a single step -- which happens under speculative decoding, where
multiple tokens (and therefore multiple stop strings) can be appended at once.
"""

import pytest

from vllm.v1.engine.detokenizer import check_stop_strings


@pytest.mark.parametrize("stop", [["a", "is"], ["is", "a"]])
def test_earliest_completing_stop_wins_regardless_of_list_order(stop):
    # " The user is a": " is a" (5 chars) was appended in one step. Both "is"
    # (index 10) and " a" (index 13) land in the same window. "is" completes
    # earlier in the text, so it must win over list order.
    text = " The user is a"
    new_char_count = len(" is a")

    assert check_stop_strings(text, new_char_count, stop, include_in_output=False) == (
        "is",
        10,
    )


@pytest.mark.parametrize("stop", [["a", "is"], ["is", "a"]])
def test_earliest_completing_stop_include_in_output(stop):
    text = " The user is a"
    new_char_count = len(" is a")

    # Truncate to the end of "is" (index 12) -> " The user is".
    assert check_stop_strings(text, new_char_count, stop, include_in_output=True) == (
        "is",
        12,
    )


def test_completion_position_not_start_position():
    # "b" starts later than "abc" but completes earlier, so it must win.
    text = "abc"
    assert check_stop_strings(
        text, len(text), ["abc", "b"], include_in_output=False
    ) == ("b", 1)


@pytest.mark.parametrize(
    "stop,expected",
    [
        (["ab", "b"], ("ab", 0)),
        (["b", "ab"], ("b", 1)),
    ],
)
def test_ties_broken_by_list_order(stop, expected):
    # "ab" and "b" both complete at index 2; list order decides the winner.
    text = "ab"
    assert (
        check_stop_strings(text, len(text), stop, include_in_output=False) == expected
    )


def test_single_stop_in_window_unchanged():
    # The common case (one stop in the window) is unaffected by the change.
    text = "hello world."
    assert check_stop_strings(text, 1, ["."], include_in_output=False) == (".", 11)
    # Stop completes at the very end -> no truncation needed (-1).
    assert check_stop_strings(text, 1, ["."], include_in_output=True) == (".", -1)


def test_no_match_and_empty_inputs_return_none():
    assert check_stop_strings("hello", 5, ["zzz"], include_in_output=False) is None
    assert check_stop_strings("hello", 0, ["h"], include_in_output=False) is None
    assert check_stop_strings("hello", 5, [], include_in_output=False) is None


from vllm.v1.engine.detokenizer import (
    PROMPT_SUFFIX_PRIME_WINDOW,
    prompt_suffix_to_prime,
)


class _FakeTokenizer:
    def __init__(self, mapping: dict[int, bytes], special_ids: set[int] | None = None):
        self.mapping = mapping
        self.special_ids = special_ids or set()

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        raw = b"".join(
            self.mapping[i]
            for i in ids
            if not (skip_special_tokens and i in self.special_ids)
        )
        return raw.decode("utf-8", errors="replace")


def test_prompt_suffix_to_prime_bounds_and_utf8_alignment():
    # Token 10 is ASCII "x"; tokens 20, 21, 22 form 3-byte UTF-8 "€" (e2 82 ac);
    # token 99 is a special token.
    tok = _FakeTokenizer(
        {10: b"x", 20: b"\xe2", 21: b"\x82", 22: b"\xac", 99: b""},
        special_ids={99},
    )

    short_prompt = [10] * PROMPT_SUFFIX_PRIME_WINDOW
    assert prompt_suffix_to_prime(tok, short_prompt, True) == short_prompt

    long_ascii = [10] * 100
    primed = prompt_suffix_to_prime(tok, long_ascii, True)
    assert primed is not None and len(primed) == PROMPT_SUFFIX_PRIME_WINDOW

    # Place a 3-token UTF-8 character across the -16 boundary so -16 starts mid-char
    # at token 21 (b"\x82" -> "\ufffd"), requiring adjust=1 to include token 20.
    split_boundary = [10] * 50 + [20, 21, 22] + [10] * (PROMPT_SUFFIX_PRIME_WINDOW - 2)
    primed_aligned = prompt_suffix_to_prime(tok, split_boundary, True)
    assert primed_aligned == split_boundary[-(PROMPT_SUFFIX_PRIME_WINDOW + 1) :]
    assert not tok.decode(primed_aligned, True).startswith("\ufffd")

    # If trailing 19 tokens (16 + adjust 0..3) are skipped special tokens, widen window to 32.
    special_tail = [10] * 50 + [99] * (PROMPT_SUFFIX_PRIME_WINDOW + 3)
    primed_special = prompt_suffix_to_prime(tok, special_tail, True)
    assert primed_special is not None and len(primed_special) == 32

    # If the prompt itself ends mid-codepoint, fall back to full prompt_token_ids.
    incomplete_end = [10] * 50 + [20, 21]
    assert prompt_suffix_to_prime(tok, incomplete_end, True) == incomplete_end
