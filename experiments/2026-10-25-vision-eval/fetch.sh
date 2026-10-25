#!/usr/bin/env bash
# Download the evaluation's benchmark files into RAW_DIR/<dataset>/, one empty directory
# each, with the node's Hugging Face login (~/.local/bin/hf).
set -euo pipefail
raw=$1
hf=$HOME/.local/bin/hf
get() {  # dataset repo file
  mkdir -p "$raw/$1"
  "$hf" download "$2" "$3" --repo-type dataset --local-dir "$raw/$1/hf" > /dev/null
  mv "$raw/$1/hf/$3" "$raw/$1/"
  rm -rf "$raw/$1/hf"
}
get ocrbench echo840/OCRBench data/test-00000-of-00001.parquet
get docvqa lmms-lab-encoder/DocVQA DocVQA/validation-00000-of-00006.parquet
get chartqa lmms-lab-encoder/ChartQA data/test-00000-of-00001.parquet
get textvqa lmms-lab-encoder/textvqa data/validation-00000-of-00003.parquet
get realworldqa xai-org/RealworldQA data/test-00000-of-00002.parquet
du -sh "$raw"/*
