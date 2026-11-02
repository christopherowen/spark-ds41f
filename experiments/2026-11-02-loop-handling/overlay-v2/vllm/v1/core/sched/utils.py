# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections import Counter
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus

# Output tokens a forced end of reasoning may take to land: the rest of the
# step that found the loop and the step already in flight, each up to its
# verification width (drafts and copy drafts).
FORCED_REASONING_END_GRACE = 64
# Output tokens a loop in a healthy answer must persist past its first
# detection before the request stops. Token soup (one token dominating the
# answer's last OUTPUT_LOOP_WINDOW tokens in many short runs, the collapse
# seen in production) stops soon. Any other loop stops only long after
# legitimate periodic output (a zero-filled array, a matrix, a table rule)
# would have ended: a 256-zero C array is 780 tokens, a 16x16 matrix 792. And
# the longest gap between detections that still counts as the same loop.
OUTPUT_SOUP_PERSIST = 256
OUTPUT_LOOP_PERSIST = 2048
OUTPUT_LOOP_GAP = 64
# ... and a loop that is not token soup must then leave fewer than one
# distinct token in OUTPUT_LOOP_MAX_DISTINCT_INV among the answer's last
# OUTPUT_LOOP_WINDOW tokens.
OUTPUT_LOOP_WINDOW = 512
OUTPUT_LOOP_MAX_DISTINCT_INV = 10


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def _find_windowed_repeating_pattern(
    token_ids: Sequence[int],
    min_pattern_size: int,
    max_pattern_size: int,
    min_count: int,
    start_idx: int = 0,
    window_size: int = 160,
    min_density: float = 0.6,
) -> tuple[int, int] | None:
    """Return ``(pattern_len, count)`` if the tail of ``token_ids`` ends with a
    phrase that repeats at least ``min_count`` times within the last
    ``window_size`` tokens (with at least one contiguous ``A + A`` pair so
    parallel list enumerations do not false-positive) and covers at least
    ``min_density`` of the span from its first matched occurrence to the end.
    """
    seq = getattr(token_ids, "_x", token_ids)
    n = len(seq)
    avail = n - start_idx
    eff_min = max(min_pattern_size, 5)
    if avail < eff_min * min_count or eff_min > max_pattern_size:
        return None

    win_start = max(start_idx, n - window_size)
    last_tok = seq[n - 1]
    checked_patterns = 0

    for i in range(n - 2, win_start - 1, -1):
        if seq[i] != last_tok:
            continue
        gap = n - 1 - i
        if gap < eff_min:
            continue
        lcs = 1
        while (
            lcs <= max_pattern_size
            and i - lcs >= start_idx
            and seq[i - lcs] == seq[n - 1 - lcs]
        ):
            lcs += 1
        if lcs < eff_min:
            continue

        phrase = seq[n - lcs : n]
        cand_lens = [lcs]
        for b in range(1, lcs - eff_min + 1):
            if phrase[:b] == phrase[lcs - b :]:
                cand_lens.append(lcs - b)

        for cand_len in cand_lens:
            if cand_len * min_count > avail:
                continue
            cand = seq[n - cand_len : n]
            count = 1
            contiguous_pairs = 0
            pos = n - cand_len
            while pos - cand_len >= win_start:
                nxt = pos - 1
                while nxt >= win_start and seq[nxt] != last_tok:
                    nxt -= 1
                if nxt - cand_len + 1 < win_start:
                    break
                if seq[nxt - cand_len + 1 : nxt + 1] == cand:
                    if nxt == pos - 1:
                        contiguous_pairs += 1
                    count += 1
                    pos = nxt - cand_len + 1
                    if (
                        count >= min_count
                        and contiguous_pairs >= 1
                        and count * cand_len >= (n - pos) * min_density
                    ):
                        return (cand_len, count)
                else:
                    pos = nxt

        checked_patterns += 1
        if checked_patterns >= 4:
            break

    return None


def _has_windowed_repeating_pattern(
    token_ids: Sequence[int],
    min_pattern_size: int,
    max_pattern_size: int,
    min_count: int,
    start_idx: int = 0,
    window_size: int = 160,
    min_density: float = 0.6,
) -> bool:
    return (
        _find_windowed_repeating_pattern(
            token_ids,
            min_pattern_size=min_pattern_size,
            max_pattern_size=max_pattern_size,
            min_count=min_count,
            start_idx=start_idx,
            window_size=window_size,
            min_density=min_density,
        )
        is not None
    )


def _find_dominant_token_window(
    token_ids: Sequence[int],
    start_idx: int = 0,
    window_size: int = 64,
    min_window: int = 32,
) -> tuple[int, int] | None:
    """Return ``(distinct, dominant_count)`` if one token makes up at least
    60% of the last ``window_size`` tokens (at least ``min_window`` of them)
    and the window holds at most one distinct token per eight.

    This is token soup: one token repeated with a few interjections, such as
    ``11x Vila, Estud compar, 9x Vila, Estud compar, 9x Vila``. Every run is
    shorter than the contiguous rules need and the period is irregular, so
    neither the contiguous nor the windowed phrase rule sees it. A run of two
    or more alternating tokens (bit strings, ``0, 0, 0``) has no token at 60%.
    """
    seq = getattr(token_ids, "_x", token_ids)
    n = len(seq)
    wlen = min(window_size, n - start_idx)
    if wlen < min_window:
        return None
    # Soup repeats its token within any eight; prose and code rarely do, and
    # this keeps the per-token cost of the count off them.
    if len(set(seq[n - 8 : n])) > 5:
        return None
    counts: dict[int, int] = {}
    for tok in seq[n - wlen : n]:
        counts[tok] = counts.get(tok, 0) + 1
    if len(counts) * 8 > wlen:
        return None
    dominant = max(counts.values())
    if dominant * 5 < wlen * 3:
        return None
    return (len(counts), dominant)


def detect_thinking_loop(
    token_ids: Sequence[int],
    start_idx: int = 0,
) -> tuple[str, int, int] | None:
    """Detect degenerate repetition loops and return ``(mode, pattern_len, count)``.

    Catches:
    1. Short contiguous token loops (pattern length 1..3 repeated >= 10 times
      with total repeated span >= 24 tokens).
    2. Phrase/sentence contiguous loops (pattern length 4..64 repeated >= 4
       times contiguously with total repeated span >= 24 tokens).
    3. Interspersed/windowed phrase loops (pattern length 5..64 repeated >= 5
       times within a 160-token window with >= 1 contiguous pair and covering
       >= 60% of the span).
    4. Token soup (one token >= 60% of the last 32..64 tokens, at most one
       distinct token per eight); ``pattern_len`` is the number of distinct
       tokens and ``count`` the dominant token's count.
    """
    seq = getattr(token_ids, "_x", token_ids)
    avail = len(seq) - start_idx
    if avail < 24:
        return None

    for pattern_len in range(1, 4):
        min_count = max(10, (24 + pattern_len - 1) // pattern_len)
        if pattern_len * min_count <= avail and _has_repeating_pattern(
            seq, pattern_len, min_count
        ):
            return ("contiguous_short", pattern_len, min_count)

    max_len = min(64, avail // 4)
    for pattern_len in range(4, max_len + 1):
        min_count = max(4, (24 + pattern_len - 1) // pattern_len)
        if pattern_len * min_count <= avail and _has_repeating_pattern(
            seq, pattern_len, min_count
        ):
            return ("contiguous_phrase", pattern_len, min_count)

    win_match = _find_windowed_repeating_pattern(
        seq,
        min_pattern_size=5,
        max_pattern_size=64,
        min_count=5,
        start_idx=start_idx,
        window_size=160,
        min_density=0.6,
    )
    if win_match is not None:
        return ("windowed_phrase", win_match[0], win_match[1])

    soup_match = _find_dominant_token_window(seq, start_idx=start_idx)
    if soup_match is not None:
        return ("token_soup", soup_match[0], soup_match[1])
    return None


def has_thinking_loop(
    token_ids: Sequence[int],
    start_idx: int = 0,
) -> bool:
    """Return True if ``token_ids[start_idx:]`` ends with a repetition loop."""
    return detect_thinking_loop(token_ids, start_idx=start_idx) is not None


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
    start_idx: int = 0,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
        start_idx: Minimum starting index in token_ids to inspect.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = max(params.min_pattern_size, 1)
    min_count = params.min_count

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    seq = getattr(token_ids, "_x", token_ids)
    avail = len(seq) - start_idx
    if avail <= 0:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > avail:
            break

        if _has_repeating_pattern(seq, pattern_len, min_count):
            return True

    if max_pattern_size >= 5 and min_count >= 3:
        return _has_windowed_repeating_pattern(
            seq,
            min_pattern_size=max(min_pattern_size, 5),
            max_pattern_size=max_pattern_size,
            min_count=max(min_count, 5),
            start_idx=start_idx,
        )

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This function optimizes for the common case of removing a single item,
    using list.remove() which avoids creating a new list. For multiple items,
    it uses a list comprehension.

    Note: For a single item, the input list is modified in-place and returned.
    For multiple items, a new list is created and returned.

    Args:
        lst: The list to remove items from.
        items_to_remove: Set of items to remove from the list.

    Returns:
        Either the modified input list (for single item) or a new list
        (for multiple items) with the specified items removed. Caller should
        always use the returned value.
        Note: for single item, it modifies the input list in-place and
        returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def stops_on_next_token(request: Request, token_id: int, max_model_len: int) -> bool:
    """Whether committing ``token_id`` next stops the request by EOS, a stop
    token or a length cap: check_stop's checks that no later token changes.
    Sink rejection commits such a token as soon as it is sampled."""
    sampling_params = request.sampling_params
    assert sampling_params is not None
    return (
        token_id == sampling_params.eos_token_id
        or token_id in (sampling_params.stop_token_ids or ())
        or request.num_tokens + 1 >= max_model_len
        or request.num_output_tokens + 1 >= request.max_tokens
    )


def check_stop(
    request: Request,
    max_model_len: int,
    thinking_loop_break: bool = True,
) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # Note(arpera):
    # Order of checks is important for min_tokens
    # If you decide to change the order, first look at these PRs please:
    # 47489, 49521, and 51299
    # These PRs show some reasoning about the order of checks
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        repetition_detection.max_pattern_size <= 0
        or repetition_detection.min_count < 2
    ):
        return False

    if request.thinking_output_start >= 0:
        if request.force_thinking_end:
            # The worker forces the end of reasoning in the request's next
            # scheduled step, after the step already in flight. A forced end
            # that never arrives stops the request instead of leaving the
            # loop unchecked until max_tokens.
            if (
                request.num_output_tokens - request.force_thinking_end_at
                <= FORCED_REASONING_END_GRACE
            ):
                return False
            return _finish_repetition(request)
        start_idx = request.thinking_output_start
        loop_info: tuple[str, int, int] | None = None
        if repetition_detection is not None and check_sequence_repetition(
            request._output_token_ids,
            repetition_detection,
            start_idx=start_idx,
        ):
            loop_info = ("custom_repetition", 0, 0)
        if loop_info is None:
            loop_info = detect_thinking_loop(
                request._output_token_ids, start_idx=start_idx
            )
        if loop_info is None:
            return False
        request.last_loop_info = loop_info
        if thinking_loop_break and request.thinking_loop_breaks == 0:
            request.force_thinking_end = True
            request.force_thinking_end_at = request.num_output_tokens
            request.thinking_loop_breaks += 1
            request.copy_draft_alloc = 0
            return False
        return _finish_repetition(request)

    start_idx = request.thinking_end_output_idx
    if repetition_detection is not None and check_sequence_repetition(
        request._output_token_ids,
        repetition_detection,
        start_idx=start_idx,
    ):
        request.last_loop_info = ("custom_repetition", 0, 0)
        return _finish_repetition(request)

    if not (thinking_loop_break or request.thinking_loop_breaks > 0):
        return False
    loop_info = detect_thinking_loop(request._output_token_ids, start_idx=start_idx)
    if request.thinking_loop_breaks > 0:
        # Reasoning already looped: stop the answer at its first loop.
        if loop_info is None:
            return False
        request.last_loop_info = loop_info
        return _finish_repetition(request)

    # The visible answer of a request whose reasoning did not loop. Legitimate
    # answers repeat too (a zero-filled array, a table rule, a file written
    # through a tool call) and then move on; a degenerate loop does not. Stop
    # only a loop that persists (token soup OUTPUT_SOUP_PERSIST tokens, any
    # other loop OUTPUT_LOOP_PERSIST) and dominates the answer's last
    # OUTPUT_LOOP_WINDOW tokens, so a false detection costs nothing: a long
    # run inside otherwise varied output (a 199-token separator rule in a
    # message whose distinct-token ratio is 0.8) is never stopped.
    num_output = request.num_output_tokens
    if loop_info is None:
        if (
            request.output_loop_since >= 0
            and num_output - request.output_loop_last > OUTPUT_LOOP_GAP
        ):
            request.output_loop_since = -1
        return False
    if request.output_loop_since < 0:
        request.output_loop_since = num_output
    request.output_loop_last = num_output
    request.last_loop_info = loop_info
    persisted = num_output - request.output_loop_since
    if persisted < OUTPUT_SOUP_PERSIST:
        return False
    window = request._output_token_ids[
        max(start_idx, num_output - OUTPUT_LOOP_WINDOW) : num_output
    ]
    if _is_token_soup(window):
        return _finish_repetition(request)
    if persisted < OUTPUT_LOOP_PERSIST:
        return False
    if len(set(window)) * OUTPUT_LOOP_MAX_DISTINCT_INV >= len(window):
        return False
    return _finish_repetition(request)


def _is_token_soup(window: Sequence[int]) -> bool:
    """One token over 60% of the window, broken into many short runs (one run
    or more every 32 tokens) by interjections: the collapse seen in production.
    One long run (a separator rule) and an exactly periodic array (no token
    over half of it) are not."""
    dominant, count = Counter(window).most_common(1)[0]
    if count * 5 < len(window) * 3:
        return False
    runs = sum(
        1
        for i, token in enumerate(window)
        if token == dominant and (i == 0 or window[i - 1] != dominant)
    )
    return runs * 32 >= len(window)


def _finish_repetition(request: Request) -> bool:
    request.status = RequestStatus.FINISHED_REPETITION
    request.stop_reason = "repetition_detected"
    return True
