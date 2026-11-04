"""Strict recorded-output comparisons; no inference or performance measurement."""

import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory/model_identity.py"
spec = importlib.util.spec_from_file_location("lossless_model_identity", PATH)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"workload": {"prompt_count": 1, "top": 2},
                          "corpus": {"sequences": 1, "sha256": "fixture"},
                          "generation": {"sequences": [{"id": "a", "prompt_ids": [1],
                              "output_ids": [2], "finish_reason": "length"}]},
                          "scores": [{"id": "a", "tokens": [2], "rank": [1],
                              "logprob": [-0.1], "top_ids": [[2, 3]], "top_logprobs": [[-0.1, -2.5]]}]}
        self.candidate = copy.deepcopy(self.reference)

    def test_complete_identical_vectors_pass(self):
        self.assertTrue(identity.compare(self.reference, self.candidate)["identical"])

    def test_same_top_token_with_any_score_change_fails(self):
        self.candidate["scores"][0]["top_logprobs"][0][1] += 1e-10
        result = identity.compare(self.reference, self.candidate)
        self.assertFalse(result["identical"])
        self.assertEqual(result["score_differences"], ["a"])

    def test_signed_zero_is_distinct(self):
        self.reference["scores"][0]["logprob"] = [0.0]
        self.candidate["scores"][0]["logprob"] = [-0.0]
        self.assertFalse(identity.compare(self.reference, self.candidate)["identical"])

    def test_missing_duplicate_failed_or_truncated_scores_are_rejected(self):
        for change in ("missing", "duplicate", "failed", "truncated"):
            value = copy.deepcopy(self.reference)
            if change == "missing":
                value["scores"] = []
            elif change == "duplicate":
                value["scores"] *= 2
            elif change == "failed":
                value["scores"][0]["error"] = "request failed"
            else:
                value["scores"][0]["top_logprobs"] = []
            with self.assertRaises(ValueError, msg=change):
                identity.compare(self.reference, value)

    def test_different_corpus_or_prompt_workload_cannot_pass(self):
        self.candidate["corpus"]["sha256"] = "different"
        self.assertFalse(identity.compare(self.reference, self.candidate)["identical"])
        self.candidate = copy.deepcopy(self.reference)
        self.candidate["workload"]["top"] = 1
        self.assertFalse(identity.compare(self.reference, self.candidate)["identical"])

    def test_nonfinite_scores_are_refused(self):
        self.candidate["scores"][0]["logprob"] = [float("nan")]
        with self.assertRaises(ValueError):
            identity.compare(self.reference, self.candidate)


if __name__ == "__main__":
    unittest.main()
