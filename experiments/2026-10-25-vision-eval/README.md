# The TileLang vision tower on public benchmarks

Review, 2026-10-25: image quality is not yet validated. The ports build's image outputs
differ from candidate 4's (B12X's vision tower), and the serving check's four generated
grids show repeatability, not equivalent image understanding. Real images, OCR and
document questions must be compared before the vision replacement is promoted. The owner
chose public benchmarks with answers, downloaded on the DGX that holds the Hugging Face
login.

This run asks the same questions of the ports build (TileLang vision tower) and of
candidate 4 (B12X's), and scores both against the benchmarks' answers:

| benchmark   | split, source                                    | items | what it covers |
|-------------|--------------------------------------------------|------:|----------------|
| OCRBench    | test, `echo840/OCRBench`                         |  1000 | text recognition, scene text, document VQA, key information, handwritten math |
| DocVQA      | validation shard 0, `lmms-lab-encoder/DocVQA`    |   300 | questions on scanned documents |
| ChartQA     | test, `lmms-lab-encoder/ChartQA`                 |   300 | questions on charts, human and generated |
| TextVQA     | validation shard 0, `lmms-lab-encoder/textvqa`   |   300 | text in real photographs |
| RealWorldQA | test shard 0, `xai-org/RealworldQA`              |   383 | real-world photographs, spatial questions |

Sampled sets are seeded (extract.py), so every run asks the same items. Questions are asked
as lmms-eval asks them: at temperature 0, with thinking off, at most 64 tokens. DocVQA,
ChartQA and TextVQA append "Answer the question using a single word or phrase." Metrics
are lmms-eval's (score.py):
- OCRBench: answer contained in the response;
- DocVQA: ANLS;
- ChartQA: relaxed accuracy;
- TextVQA: VQA accuracy;
- RealWorldQA: exact match.

For each benchmark the table gives:
- both builds' means;
- how many answers differ;
- how many items score better or worse on candidate 4 than on ports;
- a bootstrap 95% interval of the difference.

- [fetch.sh](fetch.sh): downloads the parquet files on dgx1 (`~/.local/bin/hf`, the node's
  login), each benchmark in its own directory.
- [extract.py](extract.py): `uv run --with pyarrow python -I extract.py RAW OUT` writes the
  images and `items.jsonl`.
- [vision_eval.py](vision_eval.py): asks every item (eight at once). A failed request fails
  the run.
- [score.py](score.py), [tables_arms.py](tables_arms.py): scoring and the comparison.
- [vision-eval.json](vision-eval.json): the run (ports, then candidate 4).

Data lives outside the repository, on dgx1 (`~/spark-lab/vision-eval/{raw,data}`).

## Results (run `vision-eval`, 2026-10-25 07:54-08:18 CET)

Ports (vLLM tilelang-ports 4deb7d94b's package files: the TileLang vision tower) against
candidate 4 (B12X's tower), the same 2283 items, every request answered:

| benchmark   | items | ports | candidate 4 | answers differ | candidate 4 better / worse | difference, 95% interval |
|-------------|------:|------:|------------:|---------------:|---------------------------:|-------------------------:|
| ChartQA     |   300 | 86.00 |       85.67 |             14 |                      1 / 2 | -0.33 [-1.67, +0.67]     |
| DocVQA      |   300 | 94.01 |       94.74 |             27 |                      6 / 7 | +0.73 [-0.31, +2.01]     |
| OCRBench    |  1000 | 81.10 |       81.80 |            549 |                     16 / 9 | +0.70 [-0.30, +1.70]     |
| RealWorldQA |   383 | 77.02 |       77.02 |             13 |                      6 / 6 | +0.00 [-1.83, +1.83]     |
| TextVQA     |   300 | 85.67 |       86.20 |             10 |                      3 / 2 | +0.53 [-0.73, +2.00]     |
| all         |  2283 | 83.36 |       83.79 |            613 |                    32 / 26 | +0.43 [-0.14, +1.02]     |

**No benchmark separates the two towers.** Every interval of the difference (candidate 4
minus ports, scores in points) contains zero. Overall:
- 58 items score differently: 32 better on candidate 4, 26 better on ports;
- a sign test gives p ≈ 0.5;
- the point estimate is 0.4 points in candidate 4's favour, and the interval runs from
  1.0 for candidate 4 to 0.1 for ports.

Most differing answers are OCRBench's long transcriptions (549 of 1000). Where the text
differs, the score rarely does: 25 OCRBench items change score.

The TileLang tower follows DeepSeek's reference arithmetic, where B12X's does not bit for
bit. On real photographs, scene text, documents and charts, its answers are as good as
B12X's to within these samples' resolution, about one point overall.

Both arms ran at concurrency 8. The engine never queued: mean queue time 6 ms, all eight
requests running, mean time to first token 0.78 s. Each arm took about 11 minutes. A
later run can use 16, the deployment's `--max-num-seqs`. Greedy outputs do not depend on
batching, so the answers would not change.

The full table is [results/vision-eval-table.txt](results/vision-eval-table.txt).
