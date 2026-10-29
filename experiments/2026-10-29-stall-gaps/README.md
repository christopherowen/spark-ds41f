# What the short decode gaps during a long prefill are

On #650 the decode gaps that overlap a long cold prefill take two values: a long one, about
one step carrying a prefill chunk (about 2.25 s on the ring at a 16,384-token budget), and
a short one, at about half each. So their median flips between the two from repeat to
repeat (0.69-2.12 s for candidate 10 in `../2026-10-28-candidate-10-levers`, "yield,
depth"), while their mean and p90 hold. With yield and probes the median fell to about
0.33 s. The short gaps are either engine steps that carried no chunk (decode alone, between
chunk steps), or one step's output reaching a stream as two chunks. This run tells them
apart.

## The run (`stall-gaps.json`, run `stall-gaps`)

`suite.json`: the stall step at 131K, 7 of 16 decoding streams, four repeats, without probes
(`stall-7`) and with the yield-depth run's probes (`stall-7-probes`, 500 tokens, 2-14 s into
the prefill). The servebench stall step (#673) keeps each repeat's raw overlapping gaps and
the engine steps the prefill spanned, from `vllm:iteration_tokens_total` scraped as the
document is sent and at its first token, split at 1,024 tokens into steps with a chunk and
steps without. Arms on the ring: candidate 10 and candidate 10 with yield, one boot each.

## Reading it

Per repeat, compare each stream's overlapping gaps with the engine's steps:

- Gaps a stream about equal to the engine steps, and steps without a chunk about as many as
  those with: the short gaps are decode-only steps between chunks. Then the question for
  #650 is what schedules them, and their length against the baseline step.
- Gaps a stream about twice the engine steps, with few steps without a chunk: one step's
  output arrives as two chunks, and the stall is the chunk steps alone.

The raw gaps' start offsets place the short gaps against the chunk steps. With probes, the
steps without a chunk include the probes' own prefill steps under yield.

## Results

(Filled in when the run lands.)
