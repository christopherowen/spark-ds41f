# candidate-52 at tp4-switched (sp1..sp4), 2026-11-09

These are the lab's outputs from sp1's window, copied from sp1 `results/private/`.

**Gate** (two arms):
- `candidate-52-no1030`: identical temperature-0 outputs, with TP4's hashes 8e46794a/11ec073c. #968 and #969 change no output.
- `candidate-52`: deterministic, with new hashes 18d61961/d6568649. That is #1030's declared output change.

**Screen** against candidate-50 without #1009 (the recorded reference): grid, c16 decode and output, and prefill at 32K and 256K are all the same (`lab/candidate-52-screen-tp4-switched-table.txt`). #968's "paired decode not worse" holds.

**Model behaviour:**
- #1030 changes greedy output on 128/128 prompts, diverging at a median of token 26, as declared.
- Draft acceptance is unchanged: acceptance length 2.302 against 2.304.
- Its perplexity on its own greedy text is 1.31888, against candidate-50's 1.30921 on its own. The two runs score different corpora, so this is not a like-for-like quality comparison. The release qualification's quality evals measure #1030's effect.
