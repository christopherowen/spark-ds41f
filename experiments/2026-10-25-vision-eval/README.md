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
| RealWorldQA | test shard 0, `xai-org/RealworldQA`              |  ~380 | real-world photographs, spatial questions |

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
