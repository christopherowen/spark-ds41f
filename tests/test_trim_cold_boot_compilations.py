from pathlib import Path
import unittest


class TestTrimColdBootCompilations(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.patch_path = (
            self.root
            / "patches"
            / "vllm"
            / "0045-tilelang-trim-cold-boot-compilations.patch"
        )
        self.series_path = self.root / "patches" / "vllm" / "series"

    def test_patch_is_registered_in_series(self) -> None:
        self.assertTrue(self.patch_path.is_file())
        series = [
            line.strip()
            for line in self.series_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn(self.patch_path.name, series)

    def test_patch_trims_unreachable_moe_router_and_markov_variants(self) -> None:
        patch_text = self.patch_path.read_text(encoding="utf-8")
        # Drafter 128-expert router gate split-K threshold matches the 384-expert target.
        self.assertIn("(128, 5120): 256", patch_text)
        # TileKernelsRouter declares int64 indices_dtype so TileLangExperts skips int32 dispatches.
        self.assertIn("indices_dtype = torch.int64", patch_text)
        # MoeDispatchV41Plan filters parts by min_routes / max_routes and weighted combine flag.
        self.assertIn("min_routes: int = 1", patch_text)
        self.assertIn("max_routes: int | None = None", patch_text)
        self.assertIn("weighted: bool = False", patch_text)
        # MoeExpertsV41Plan filters prefill tiles by max_rows.
        self.assertIn("def _has_prefill_tiles(self) -> bool:", patch_text)
        # Markov transition head bounds token tiles by max_num_seqs.
        self.assertIn("self.markov_head.markov_w2.tilelang_max_rows", patch_text)
        self.assertIn("def _head_row_tiles(layer: nn.Module)", patch_text)
        # prepare_declared marks compiled callables so registry.warmup() does not re-expand them.
        self.assertIn("self._prepared_calls: set[int] = set()", patch_text)


if __name__ == "__main__":
    unittest.main()
