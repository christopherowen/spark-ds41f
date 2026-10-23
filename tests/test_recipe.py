"""One recipe for every shape: config/cluster.json resolves to the concrete
configurations that used to be separate files, and the shapes differ only where
the cabling or the memory per rank differs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

ARCHIVE = ROOT / "experiments" / "2026-10-14-one-recipe"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


BUDGET_KEYS = ("adaptive_verification_cost_scale", "adaptive_verification_row_cost")


def serving_arguments(cluster: dict) -> list[str]:
    """``serve_args`` without what selects the checkpoint and its sampling defaults (the
    model path and ``--override-generation-config``) and without the draft budget's
    constant (r6x replaced the boot-timed cost model's scale with a row cost)."""
    args = cluster["serve_args"][1:]
    if "--override-generation-config" in args:
        at = args.index("--override-generation-config")
        del args[at : at + 2]
    if "--speculative-config" in args:
        at = args.index("--speculative-config") + 1
        speculative = json.loads(args[at])
        args[at] = json.dumps({k: v for k, v in speculative.items() if k not in BUDGET_KEYS})
    return args


CONTEXT_ARGS = ("--max-model-len", "--kv-cache-memory-bytes")


def argument(cluster: dict, flag: str) -> str:
    args = cluster["serve_args"]
    return args[args.index(flag) + 1]


def without_context(args: list[str]) -> list[str]:
    """``args`` without the context length and KV pool, the sizing r6x's TP3 1M changed."""
    out = list(args)
    for flag in CONTEXT_ARGS:
        if flag in out:
            at = out.index(flag)
            del out[at : at + 2]
    return out


def draft_budget(cluster: dict) -> dict:
    args = cluster["serve_args"]
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    return {k: v for k, v in speculative.items() if k in BUDGET_KEYS}


def load_spark():
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader("spark", loader)
    spark = importlib.util.module_from_spec(spec)
    loader.exec_module(spark)
    return spark


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = load("config/cluster.json")

    def test_tp4_ring_was_r6ks_production_file(self):
        """The unification changed nothing for TP4: r6k's recipe, frozen at r6n's
        promotion, resolves to r6k's file."""
        resolved = recipe.resolve(load("experiments/2026-10-15-r6n/r6k/cluster.json"), "tp4-ring")
        expected = dict(load("experiments/2026-10-14-one-recipe/tp4-r6k-production.json"),
                        shape="tp4-ring", schema_version=1)
        self.assertEqual(resolved, expected)

    def test_tp4_ring_is_the_released_r6n_arm(self):
        """Promotion runs what the release measured: tp4-ring, frozen at r6s's TP3-only
        promotion, resolves to r6n's release arm, apart from the baseline, lock and
        reference that name it."""
        resolved = recipe.resolve(load("experiments/2026-10-16-r6s/r6n/cluster.json"), "tp4-ring")
        released = load_spark().read_cluster("experiments/2026-10-15-r6n/r6n-release.json")
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape")
        self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                         {k: v for k, v in released.items() if k not in naming})

    def test_each_shape_is_its_1022_arm(self):
        """Each shape resolves to spark-ds41f-1022's arm at that shape, apart from the names (the
        arms extend the recipe and name the release's experiment lock)."""
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")
        for shape in ("tp4-ring", "tp3-triangle"):
            arm = f"experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-{shape.split('-')[0]}.json"
            resolved = recipe.resolve(self.recipe, shape)
            released = load_spark().read_cluster(arm)
            self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                             {k: v for k, v in released.items() if k not in naming}, shape)

    def test_r6x_frozen_at_1022_is_its_arms(self):
        """r6x's recipe, frozen at spark-ds41f-1022's promotion, resolves at each shape to r6x's
        arm there: its release arm on the ring (which extends the frozen r6v recipe) and its 1M
        arm on the triangle."""
        frozen = load("experiments/2026-10-22-spark-ds41f-1022/r6x/cluster.json")
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")
        for shape, arm in (("tp4-ring", "experiments/2026-10-18-r6x/r6x-tp4-1017.json"),
                           ("tp3-triangle", "experiments/2026-10-19-r6x-tp3-1m/r6x-tp3-1m-1017.json")):
            resolved = recipe.resolve(frozen, shape)
            released = load_spark().read_cluster(arm)
            self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                             {k: v for k, v in released.items() if k not in naming}, shape)

    def test_the_model_profile_mounts_the_selected_checkpoint(self):
        """Every shape mounts the recipe's model (1017: the hub, since its blobs link on into
        the hub's shared store, the snapshot served from inside it); an arm that sets
        another model serves it, whatever mounts it lists."""
        self.assertEqual(self.recipe["model"], "v41-flash-1017")
        profile = self.recipe["models"]["v41-flash-1017"]
        for shape in ("tp3-triangle", "tp4-ring"):
            resolved = recipe.resolve(self.recipe, shape)
            self.assertEqual(resolved["container"]["mounts"][0], [profile["hub"], "/hub", "ro"])
            self.assertEqual(resolved["serve_args"][0], f"/hub/{profile['snapshot']}")
        base = recipe.resolve(self.recipe, "tp3-triangle")
        self.assertEqual(recipe.model_snapshot(base), f"{profile['hub']}/{profile['snapshot']}")
        # The original checkpoint: snapshot and blobs mounted, served from /models.
        arm = recipe.overlay({"extends": {}, "set": {"model": "v41-flash"}}, base)
        original = self.recipe["models"]["v41-flash"]
        mounts = arm["container"]["mounts"]
        self.assertEqual(mounts[:2], [[original["snapshot"], "/models", "ro"],
                                      [original["blobs"], "/blobs", "ro"]])
        self.assertEqual(mounts[2:], base["container"]["mounts"][1:])
        self.assertEqual(arm["serve_args"][0], "/models")
        self.assertEqual(arm["serve_args"][1:], base["serve_args"][1:])
        # And back: an arm over it that sets 1017 serves from the hub again.
        back = recipe.overlay({"extends": {}, "set": {"model": self.recipe["model"]}}, arm)
        self.assertEqual(back["container"]["mounts"], base["container"]["mounts"])
        self.assertEqual(back["serve_args"], base["serve_args"])
        with self.assertRaises(SystemExit):
            recipe.overlay({"extends": {}, "set": {"model": "nonexistent"}}, base)
        # An arm that lists its own mounts (an overlay of changed files) still serves its
        # model: the list replaces the base's, so the model's mounts are put back first.
        extra = ["{home}/overlay/moe.py", "/opt/spark3/candidate/vllm/moe.py", "ro"]
        listed = recipe.overlay({"extends": {}, "set": {"container": {"mounts": [
            ["{home}/projects/spark-ds41f/cache", "/cache", "rw"], extra]}}}, base)
        self.assertEqual(listed["container"]["mounts"],
                         [[profile["hub"], "/hub", "ro"], ["{home}/projects/spark-ds41f/cache", "/cache", "rw"],
                          extra])
        self.assertEqual(listed["serve_args"], base["serve_args"])

    def test_generation_defaults_render_as_override_generation_config(self):
        """The recipe's sampling defaults (DeepSeek's for 1017) render once, beside
        --generation-config; an arm merges its own values into them."""
        self.assertEqual(self.recipe["generation_defaults"], {"temperature": 1.0, "top_p": 1.0})
        arm = recipe.resolve(self.recipe, "tp3-triangle")
        args = arm["serve_args"]
        at = args.index("--generation-config")
        self.assertEqual(args[at + 2 : at + 4],
                         ["--override-generation-config", '{"temperature":1.0,"top_p":1.0}'])
        self.assertEqual(args.count("--override-generation-config"), 1)
        # An arm over that one merges its values (as every overlay merges a dictionary)
        # and replaces the flag rather than repeating it.
        again = recipe.overlay({"extends": {}, "set": {"generation_defaults": {"temperature": 0.6}}}, arm)
        self.assertEqual(again["serve_args"].count("--override-generation-config"), 1)
        self.assertIn('{"temperature":0.6,"top_p":1.0}', again["serve_args"])

    def test_tp3_keeps_its_cabling_and_sizing(self):
        """TP3 keeps what the triangle and its memory need, and nothing else of r6h."""
        resolved = recipe.resolve(self.recipe, "tp3-triangle")
        old = load("experiments/2026-10-14-one-recipe/tp3-r6h-production.json")
        # Since r6x the triangle serves 1M with a 4.90625 GiB pool (the expert scale codes freed
        # the memory); everything else of r6h's sizing, drafts and capture ladder stays.
        self.assertEqual(argument(resolved, "--max-model-len"), "1048576")
        self.assertEqual(argument(resolved, "--kv-cache-memory-bytes"), "5268045824")
        self.assertEqual(without_context(serving_arguments(resolved)), without_context(serving_arguments(old)))
        self.assertEqual(resolved["fabric"], old["fabric"])
        for key, value in old["environment"].items():
            if key.startswith("NCCL_") or key in ("SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "SPARK3_DISPLAY_CARVEOUT_WEIGHTS"):
                self.assertEqual(resolved["environment"].get(key), value, key)
        # TP3 schedules on its build's own cost curves (bin/spark derives the directory).
        self.assertNotIn("SPARK3_DSPARK_COST_DIR", resolved["environment"])
        tp4 = recipe.resolve(self.recipe, "tp4-ring")
        self.assertEqual(resolved["container"], tp4["container"])  # one image and run arguments for every shape
        # Every shape budgets drafts by the row cost; nothing is timed at boot (since r6x).
        self.assertEqual(draft_budget(resolved), {"adaptive_verification_row_cost": 0.035})
        self.assertEqual(draft_budget(tp4), draft_budget(resolved))
        # spark-ds41f-1022 was released on the triangle, then on the ring: each shape has its own run.
        self.assertEqual(resolved["benchmark_reference"], "manifests/suites/2026-10-22-spark-ds41f-1022-tp3")
        self.assertEqual(tp4["benchmark_reference"], "manifests/suites/2026-10-22-spark-ds41f-1022-tp4")
        self.assertNotIn("benchmark_reference", recipe.resolve(self.recipe, "tp3-triangle-4k"))  # not benched

    def test_only_the_cabling_and_memory_differ_between_shapes(self):
        tp3, tp4 = (recipe.resolve(self.recipe, name) for name in ("tp3-triangle", "tp4-ring"))
        different = {key for key in set(tp3["environment"]) | set(tp4["environment"])
                     if tp3["environment"].get(key) != tp4["environment"].get(key)}
        # Since r6v both shapes derive their DSpark cost curves per build: no pinned directory.
        allowed = {key for key in different if key.startswith("NCCL_")} | {
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "SPARK3_DISPLAY_CARVEOUT_WEIGHTS"}
        self.assertEqual(different, allowed)
        for name, entry in self.recipe["shapes"].items():
            self.assertTrue(entry.get("notes"), f"shape {name} says why it differs")

    def test_derived_values_follow_streams_and_drafts(self):
        self.assertEqual(recipe.capture_sizes(48), [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48])
        self.assertEqual(recipe.capture_sizes(96)[-6:], [56, 64, 72, 80, 88, 96])
        values = recipe.values(self.recipe, recipe.shape_entry(self.recipe, "tp4-ring"))
        self.assertEqual(values["max_cudagraph_capture_size"], "96")
        self.assertEqual(values["long_prefill_token_threshold"], "8096")
        self.assertEqual(values["tensor_parallel_size"], "4")

    def test_each_shape_pins_its_prefill_budget(self):
        """Each shape's long-prefill budget is its own sweep's answer (a TP3 rank holds a
        third of the model, so its headroom differs): changing one must not move another.
        The threshold is the budget less streams x (1 + drafts)."""
        expected = {"tp4-ring": ("8192", "8096"), "tp3-triangle": ("4096", "4048"),
                    "tp3-triangle-4k": ("4096", "4048")}
        self.assertEqual(set(self.recipe["shapes"]), set(expected))
        for name, budget in expected.items():
            values = recipe.values(self.recipe, recipe.shape_entry(self.recipe, name))
            self.assertEqual((values["max_num_batched_tokens"], values["long_prefill_token_threshold"]),
                             budget, name)

    def test_a_variant_is_qualified_on_its_own(self):
        resolved = recipe.resolve(self.recipe, "tp3-triangle-4k")
        old = load("experiments/2026-10-14-one-recipe/tp3-4k-r6h.json")
        self.assertEqual(resolved["host"]["kernel"], old["host"]["kernel"])
        self.assertEqual(serving_arguments(resolved), serving_arguments(old))
        self.assertNotIn("benchmark_reference", resolved)
        qualified = dict(self.recipe)
        qualified["shapes"] = dict(self.recipe["shapes"])
        qualified["shapes"]["tp3-triangle"] = dict(self.recipe["shapes"]["tp3-triangle"], benchmark_reference="x.json")
        self.assertNotIn("benchmark_reference", recipe.resolve(qualified, "tp3-triangle-4k"))

    def test_a_site_that_names_no_shape_gets_the_recipes_default(self):
        """tp4-ring is the default; the site's config/site.json, then --shape, override it."""
        self.assertEqual(self.recipe["default_shape"], "tp4-ring")
        self.assertIn(self.recipe["default_shape"], self.recipe["shapes"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(recipe.chosen_shape(self.recipe, root), "tp4-ring")
            (root / "config").mkdir()
            (root / recipe.SITE_FILE).write_text(json.dumps({"shape": "tp3-triangle"}))
            self.assertEqual(recipe.chosen_shape(self.recipe, root), "tp3-triangle")
            self.assertEqual(recipe.chosen_shape(self.recipe, root, "tp3-triangle-4k"), "tp3-triangle-4k")

    def test_the_helper_resolves_the_default_shape_when_the_site_names_none(self):
        spark = load_spark()
        with mock.patch.object(spark.cluster_recipe, "site_shape", return_value=None):
            self.assertEqual(spark.read_cluster("config/cluster.json")["shape"], "tp4-ring")

    def test_the_default_is_not_part_of_a_resolved_configuration(self):
        for name in self.recipe["shapes"]:
            self.assertNotIn("default_shape", recipe.resolve(self.recipe, name))

    def test_shapes_hold_only_shape_settings(self):
        bad = dict(self.recipe)
        bad["shapes"] = {"odd": {"nodes": 3, "fabric": {}, "sizing": {}, "container": {}}}
        with self.assertRaises(SystemExit):
            recipe.resolve(bad, "odd")
        repeated = dict(self.recipe)
        repeated["shapes"] = dict(self.recipe["shapes"])
        repeated["shapes"]["tp4-ring"] = dict(self.recipe["shapes"]["tp4-ring"],
                                              environment={"VLLM_L2_PREFETCH_GRID": "2"})
        with self.assertRaises(SystemExit):
            recipe.resolve(repeated, "tp4-ring")
        with self.assertRaises(SystemExit):
            recipe.resolve(self.recipe, "tp5-star")

    def test_no_other_cluster_files(self):
        """One recipe: no per-shape copies beside it."""
        self.assertEqual(sorted(p.name for p in (ROOT / "config").glob("cluster*.json")), ["cluster.json"])



class OverlayTest(unittest.TestCase):
    """Experiment arms hold only what they change over the configuration they extend."""

    def test_an_overlay_sets_unsets_and_is_not_production(self):
        base = recipe.resolve(load("config/cluster.json"), "tp4-ring")
        arm = {"extends": {"config": "config/cluster.json", "shape": "tp4-ring"},
               "set": {"container": {"image": "x:y"}, "environment": {"NEW": "1"}},
               "unset": ["environment.SPARKNET_ROCE_TIMEOUT_MS"]}
        resolved = recipe.overlay(arm, base)
        self.assertEqual(resolved["container"]["image"], "x:y")
        self.assertEqual(resolved["container"]["name"], base["container"]["name"])  # merged, not replaced
        self.assertEqual(resolved["environment"]["NEW"], "1")
        self.assertNotIn("SPARKNET_ROCE_TIMEOUT_MS", resolved["environment"])
        self.assertNotIn("shape", resolved)
        self.assertNotIn("benchmark_reference", resolved)
        with self.assertRaises(SystemExit):
            recipe.overlay({"extends": {}, "unset": ["environment.NOT_THERE"]}, base)
        with self.assertRaises(SystemExit):
            recipe.overlay({"extends": {}, "container": {}}, base)

    def test_r6m_changes_only_its_own_settings_over_r6l(self):
        spark = load_spark()
        r6l = load("experiments/2026-10-14-r6l/r6l-release.json")
        r6m = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        self.assertEqual({k for k in set(r6l["environment"]) | set(r6m["environment"])
                          if r6l["environment"].get(k) != r6m["environment"].get(k)},
                         {"SPARKNET_ROCE_KERNELS", "SPARKNET_ROCE_SPIN_LIMIT", "SPARKNET_ROCE_TIMEOUT_MS",
                          "SPARK3_ENGRAM_ASYNC"})
        self.assertEqual(r6m["serve_args"], r6l["serve_args"])
        self.assertEqual({k: v for k, v in r6m.items() if k not in ("container", "environment", "upstreams_config")},
                         {k: v for k, v in r6l.items() if k not in ("container", "environment", "upstreams_config")})
        # An arm of the recipe names its shape: it must not follow the site's cabling.
        unpinned = {"extends": {"config": "config/cluster.json"}, "set": {}}
        read = spark.read_json
        with mock.patch.object(spark, "read_json", lambda path: unpinned if path == "arm.json" else read(path)), \
                self.assertRaisesRegex(SystemExit, "name its shape"):
            spark.read_cluster("arm.json")


if __name__ == "__main__":
    unittest.main()
