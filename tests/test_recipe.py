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
# Flags that change only what a response reports, added after the archived records.
REPORTING_ARGS = ("--enable-prompt-tokens-details",)
# The JIT monitor installs hooks and reports compiles; it computes nothing. spark-ds41f-1025
# turns it off: under TileLang it has nothing left to report.
MONITOR_ARGS = ("--jit-monitor-mode",)


def serving_arguments(cluster: dict) -> list[str]:
    """``serve_args`` without what selects the checkpoint and its sampling defaults (the
    model path and ``--override-generation-config``), without the draft budget's
    constant (r6x replaced the boot-timed cost model's scale with a row cost),
    without the reporting flags (#455 added the prompt-cache report) and without the JIT
    monitor's mode (off since spark-ds41f-1025)."""
    args = [arg for arg in cluster["serve_args"][1:] if arg not in REPORTING_ARGS]
    for flag in MONITOR_ARGS:
        if flag in args:
            at = args.index(flag)
            del args[at : at + 2]
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


def renamed(cluster: dict) -> dict:
    """A resolved profile of spark-ds41f-1022 under the names of the release after it."""
    out = json.loads(json.dumps(cluster))
    out["container"]["name"] = {"dsv41-karmic-kraken": "spark-ds41f"}[out["container"]["name"]]
    out["container"]["expected_labels"] = {
        key.replace("local.spark3.", "local.spark-ds41f.", 1): value
        for key, value in out["container"]["expected_labels"].items()}
    environment = {}
    for key, value in out["environment"].items():
        if key in ("PYTHONPATH", "SPARK3_DSPARK_PROFILE_TOKENS"):
            continue
        value = value.replace("/cache/kkref/jit/vllm-r5o", "/cache/spark-ds41f/jit/vllm")
        environment[key.replace("SPARK3_", "SPARK_DS41F_", 1)] = value.replace("/cache/kkref", "/cache/spark-ds41f")
    out["environment"] = environment
    return out


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

    # The 21 settings for kernels serving does not run, which spark-ds41f-1025 removes
    # (experiments/2026-10-24-tilelang-completion, "Startup without unused kernels").
    UNUSED_KERNEL_SETTINGS = (
        "B12X_AUTOTUNE", "B12X_BIND_COMPILE_WORKERS", "B12X_COMPILE_CACHE_DIR", "B12X_CUTE_COMPILE_CACHE_DIR",
        "B12X_DENSE_SPLITK_TURBO", "B12X_STATE_COMPILE_WORKERS", "B12X_W4A8_TINY_DECODE",
        "B12X_WEIGHTS_COMPILE_WORKERS", "CUTE_DSL_ARCH", "CUTE_DSL_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
        "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_AOT_COMPILE", "VLLM_USE_MEGA_AOT_ARTIFACT",
        "VLLM_USE_STANDALONE_COMPILE", "TORCHINDUCTOR_CACHE_DIR", "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        "VLLM_DS41_ATTENTION_COMPUTE", "VLLM_DS41_DRAFT_NVFP4_HEAD", "VLLM_DS41_MARKOV_NVFP4",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS")

    def test_each_shape_is_its_1022_arm_renamed(self):
        """Each shape resolves to spark-ds41f-1022's arm at that shape under the names of the
        release after it (the owner's decision of 2026-10-19), on spark-ds41f-1025's image, and
        nothing else: the container, label prefix, variables and cache paths are renamed,
        PYTHONPATH is left to the image (which sets it to the same trees) and the boot-profile
        variable nothing reads since vLLM 0184 is gone. The release also changes five things on
        purpose: the ring's long-prefill budget is 16,384 (the owner, 2026-10-23, "do 16";
        experiments/2026-10-23-next-release, budget sweeps), glibc malloc keeps two arenas and a
        fixed 1 MiB mmap threshold (#439's growth; the memory window), every response reports its
        prompt-cache hits (#469), the JIT monitor is off, and the 21 settings for kernels serving
        does not run are gone (experiments/2026-10-24-tilelang-completion)."""
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")
        self.assertEqual(len(self.UNUSED_KERNEL_SETTINGS), 21)
        for shape in ("tp4-ring", "tp3-triangle"):
            arm = f"experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-{shape.split('-')[0]}.json"
            resolved = recipe.resolve(self.recipe, shape)
            released = renamed(load_spark().read_cluster(arm))
            released["environment"] = {**{k: v for k, v in released["environment"].items()
                                          if k not in self.UNUSED_KERNEL_SETTINGS},
                                       "MALLOC_ARENA_MAX": "2", "MALLOC_MMAP_THRESHOLD_": "1048576"}
            released["container"]["image"] = "spark-ds41f:19f2c20ed4d6-next-6"
            released["container"]["expected_labels"]["local.spark-ds41f.vllm.tree"] = (
                "6f8e062f0cfb5b5fe016df9ef965499ed5ea8519")
            # The prompt-cache report (#469), where the recipe renders it.
            args = released["serve_args"] = list(released["serve_args"])
            args.insert(resolved["serve_args"].index("--enable-prompt-tokens-details"),
                        "--enable-prompt-tokens-details")
            args[args.index("--jit-monitor-mode") + 1] = "off"
            if shape == "tp4-ring":
                for flag, value in (("--max-num-batched-tokens", "16384"),
                                    ("--long-prefill-token-threshold", "16288")):
                    args[args.index(flag) + 1] = value
            self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                             {k: v for k, v in released.items() if k not in naming}, shape)
        self.assertEqual(self.recipe["container"]["image"], "spark-ds41f:19f2c20ed4d6-next-6")

    def test_the_ring_is_candidate_6s_arm(self):
        """Promotion runs what the release measured: tp4-ring resolves to candidate 6's arm,
        apart from the baseline, lock and reference that name it."""
        resolved = recipe.resolve(self.recipe, "tp4-ring")
        released = load_spark().read_cluster("experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json")
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")
        self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                         {k: v for k, v in released.items() if k not in naming})

    def test_the_recipe_keeps_no_old_live_name(self):
        """No old name is left."""
        text = json.dumps(self.recipe)
        for old in ("spark3", "SPARK3_", "kkref", "karmic", "dsv41-", "vllm-r5o", "PYTHONPATH"):
            self.assertNotIn(old, text)
        self.assertEqual(self.recipe["container"]["name"], "spark-ds41f")
        self.assertTrue(all(key.startswith("local.spark-ds41f.") for key in self.recipe["container"]["expected_labels"]))

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

    def test_1022_frozen_for_the_next_release_is_its_arms(self):
        """spark-ds41f-1022's recipe, frozen for the next release (which renames the container,
        labels, paths and variables), resolves at each shape to its release arm, and the
        previous release's and the next release's arms over image -1022-4 or -next-1 extend
        the frozen copy, so they keep booting those images under the names they read."""
        frozen_path = "experiments/2026-10-23-next-release/1022/cluster.json"
        frozen = load(frozen_path)
        self.assertEqual(frozen["upstreams_config"], "experiments/2026-10-23-next-release/1022/upstreams.lock.json")
        naming = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")
        spark = load_spark()
        for shape in ("tp4-ring", "tp3-triangle"):
            arm = f"experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-{shape.split('-')[0]}.json"
            resolved = recipe.resolve(frozen, shape)
            released = spark.read_cluster(arm)
            self.assertEqual({k: v for k, v in resolved.items() if k not in naming},
                             {k: v for k, v in released.items() if k not in naming}, shape)
        arms = ["experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp4.json",
                "experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp3.json",
                "experiments/2026-10-23-next-release/candidate-tp4.json",
                "experiments/2026-10-23-next-release/rc-tp4.json",
                "experiments/2026-10-23-next-release/rc-tp3.json"]
        for arm in arms:
            self.assertEqual(load(arm)["extends"]["config"], frozen_path, arm)
        for path in ("experiments/2026-10-22-spark-ds41f-1022", "experiments/2026-10-23-next-release"):
            for file in sorted((ROOT / path).rglob("*.json")):
                if file.is_relative_to(ROOT / "experiments/2026-10-23-next-release/rc2"):
                    continue  # candidate 2 runs the renamed recipe on its own image
                data = json.loads(file.read_text())
                if isinstance(data, dict) and "extends" in data:
                    self.assertNotEqual(data["extends"].get("config"), "config/cluster.json", file)
        for arm in arms + ["experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp4-dev.json",
                           "experiments/2026-10-23-next-release/rc-tp4-dev.json"]:
            resolved = spark.read_cluster(arm)
            self.assertEqual(resolved["container"]["name"], "dsv41-karmic-kraken", arm)
            self.assertIn(resolved["container"]["image"], ("vllm-ds41f-kkref:19f2c20ed4d6-1022-4",
                                                           "vllm-ds41f-kkref:19f2c20ed4d6-next-1"), arm)
            self.assertEqual(resolved["environment"]["SPARK3_DSPARK_VOCAB_PARALLEL"], "1", arm)
            self.assertTrue(all(key.startswith("local.spark3.") for key in resolved["container"]["expected_labels"]))

    def test_candidate_6_on_the_triangle_is_its_ring_arm(self):
        """Candidate 6's triangle arm is its ring arm at tp3-triangle (one image, labels and
        lock), so the triangle qualifies the build the ring did, at the recipe's TP3 sizing.
        The triangle's budget is 6,144 from this release (owner, 2026-10-25); the ring and the
        4 KiB variant keep theirs."""
        rc6 = "experiments/2026-10-25-spark-ds41f-candidate/rc6"
        spark = load_spark()
        tp3, tp4 = (spark.read_cluster(f"{rc6}/rc-{shape}.json") for shape in ("tp3", "tp4"))
        candidate = load("experiments/2026-10-25-spark-ds41f-candidate/cluster.json")
        self.assertEqual({k: v for k, v in load(f"{rc6}/rc-tp3.json").items() if k != "notes"},
                         {**{k: v for k, v in load(f"{rc6}/rc-tp4.json").items() if k != "notes"},
                          "extends": {"config": "experiments/2026-10-25-spark-ds41f-candidate/cluster.json",
                                      "shape": "tp3-triangle"}})
        self.assertEqual(tp3["container"], tp4["container"])
        self.assertEqual(tp3["upstreams_config"], tp4["upstreams_config"])
        at_shape = recipe.resolve(candidate, "tp3-triangle")
        self.assertEqual(tp3["serve_args"], at_shape["serve_args"])
        self.assertEqual(tp3["environment"], at_shape["environment"])
        budgets = {"tp4-ring": ("16384", "16288"), "tp3-triangle": ("6144", "6096"),
                   "tp3-triangle-4k": ("4096", "4048")}
        self.assertEqual(set(candidate["shapes"]), set(budgets))
        for name, budget in budgets.items():
            values = recipe.values(candidate, recipe.shape_entry(candidate, name))
            self.assertEqual((values["max_num_batched_tokens"], values["long_prefill_token_threshold"]),
                             budget, name)
        self.assertEqual((argument(tp3, "--max-num-batched-tokens"), argument(tp3, "--long-prefill-token-threshold")),
                         budgets["tp3-triangle"])

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
        extra = ["{home}/overlay/moe.py", "/opt/spark-ds41f/candidate/vllm/moe.py", "ro"]
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
                self.assertEqual(resolved["environment"].get(key.replace("SPARK3_", "SPARK_DS41F_", 1)), value, key)
        tp4 = recipe.resolve(self.recipe, "tp4-ring")
        self.assertEqual(resolved["container"], tp4["container"])  # one image and run arguments for every shape
        # Every shape budgets drafts by the row cost; nothing is timed at boot (since r6x).
        self.assertEqual(draft_budget(resolved), {"adaptive_verification_row_cost": 0.035})
        self.assertEqual(draft_budget(tp4), draft_budget(resolved))
        # spark-ds41f-1025 was released on the ring; the triangle waits for its own run.
        self.assertNotIn("benchmark_reference", resolved)
        self.assertEqual(tp4["benchmark_reference"], "manifests/suites/2026-10-25-spark-ds41f-1025-tp4")
        self.assertNotIn("benchmark_reference", recipe.resolve(self.recipe, "tp3-triangle-4k"))  # not benched

    def test_every_shape_reports_prompt_cache_hits(self):
        """Prefix caching reuses cached blocks; --enable-prompt-tokens-details reports the reuse
        in every response's usage.prompt_tokens_details (cached_tokens). They are separate
        switches: without the second the cache works and the key is absent (#455)."""
        for name in self.recipe["shapes"]:
            args = recipe.resolve(self.recipe, name)["serve_args"]
            self.assertEqual(args.count("--enable-prefix-caching"), 1, name)
            self.assertEqual(args.count("--enable-prompt-tokens-details"), 1, name)

    def test_only_the_cabling_and_memory_differ_between_shapes(self):
        tp3, tp4 = (recipe.resolve(self.recipe, name) for name in ("tp3-triangle", "tp4-ring"))
        different = {key for key in set(tp3["environment"]) | set(tp4["environment"])
                     if tp3["environment"].get(key) != tp4["environment"].get(key)}
        allowed = {key for key in different if key.startswith("NCCL_")} | {
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS"}
        self.assertEqual(different, allowed)
        for name, entry in self.recipe["shapes"].items():
            self.assertTrue(entry.get("notes"), f"shape {name} says why it differs")

    def test_derived_values_follow_streams_and_drafts(self):
        self.assertEqual(recipe.capture_sizes(48), [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48])
        self.assertEqual(recipe.capture_sizes(96)[-6:], [56, 64, 72, 80, 88, 96])
        values = recipe.values(self.recipe, recipe.shape_entry(self.recipe, "tp4-ring"))
        self.assertEqual(values["max_cudagraph_capture_size"], "96")
        self.assertEqual(values["long_prefill_token_threshold"], "16288")
        self.assertEqual(values["tensor_parallel_size"], "4")

    def test_ending_the_ladder_at_its_rows_keeps_every_shapes_serve_args(self):
        """The capture ladder ends at streams x (1 + drafts) even when that falls between two of
        its steps (#434). Every shape's rows were a step already (96 on the ring, 48 on the
        triangles), so each shape renders exactly what it rendered before."""
        def previous_ladder(max_size):  # the ladder before #434, which could stop below max_size
            ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
            return [size for size in ladder if size <= max_size]

        triangle = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48]
        expected = {"tp4-ring": ("16", "96", triangle + [56, 64, 72, 80, 88, 96], "16288"),
                    "tp3-triangle": ("8", "48", triangle, "4048"),
                    "tp3-triangle-4k": ("8", "48", triangle, "4048")}
        self.assertEqual(set(self.recipe["shapes"]), set(expected))
        for name, (streams, largest, ladder, threshold) in expected.items():
            resolved = recipe.resolve(self.recipe, name)
            with mock.patch.object(recipe, "capture_sizes", previous_ladder):
                self.assertEqual(resolved, recipe.resolve(self.recipe, name), name)
            self.assertEqual(argument(resolved, "--max-num-seqs"), streams, name)
            self.assertEqual(argument(resolved, "--max-cudagraph-capture-size"), largest, name)
            self.assertEqual(argument(resolved, "--compilation-config"),
                             '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],'
                             f'"cudagraph_capture_sizes":{json.dumps(ladder, separators=(",", ":"))}}}', name)
            self.assertEqual(argument(resolved, "--long-prefill-token-threshold"), threshold, name)
            self.assertEqual(recipe.capture_problems(resolved), ([], []), name)

    def test_a_stream_count_between_ladder_steps_renders_a_consistent_pair(self):
        """Six streams of 1 + 5 rows are 36, between the ladder's 32 and 40: the ladder ends at
        36, as --max-cudagraph-capture-size does (#434 booted into vLLM's 36-versus-32 refusal)."""
        six = json.loads(json.dumps(self.recipe))
        six["shapes"]["tp3-triangle"]["sizing"]["max_num_seqs"] = 6
        for name in ("tp3-triangle", "tp3-triangle-4k"):
            resolved = recipe.resolve(six, name)
            ladder = json.loads(argument(resolved, "--compilation-config"))["cudagraph_capture_sizes"]
            self.assertEqual(argument(resolved, "--max-num-seqs"), "6", name)
            self.assertEqual(argument(resolved, "--max-cudagraph-capture-size"), "36", name)
            self.assertEqual(ladder, [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36], name)
            self.assertEqual(argument(resolved, "--long-prefill-token-threshold"), "4060", name)
            self.assertEqual(recipe.capture_problems(resolved), ([], []), name)
        # Every stream count from 1 to 32, at every draft count up to 7, ends where it should.
        for drafts in range(8):
            for streams in range(1, 33):
                rows = streams * (1 + drafts)
                ladder = recipe.capture_sizes(rows)
                self.assertEqual(ladder[-1], rows)
                self.assertEqual(ladder, sorted(set(ladder)))

    def test_doctor_names_the_stream_count_when_a_profile_overrides_the_ladder(self):
        """A profile that writes its own serve_args can change --max-num-seqs and leave the
        ladder behind. A ladder that does not end at --max-cudagraph-capture-size is an error
        (vLLM refuses it); one that ends below streams x (1 + drafts) is a warning, and one
        that ends above it is fine (mixed prefill steps use its larger graphs)."""
        spark = load_spark()
        base = recipe.resolve(self.recipe, "tp3-triangle")

        def profile(streams, largest, ladder):
            cluster = json.loads(json.dumps(base))
            args = cluster["serve_args"]
            args[args.index("--max-num-seqs") + 1] = str(streams)
            args[args.index("--max-cudagraph-capture-size") + 1] = str(largest)
            at = args.index("--compilation-config") + 1
            compilation = json.loads(args[at])
            compilation["cudagraph_capture_sizes"] = ladder
            args[at] = json.dumps(compilation)
            return cluster

        def capture_findings(cluster):
            nodes = load("config/nodes.example.json")
            with mock.patch.object(spark, "tracked_secret_problems", return_value=[]):
                findings = spark.local_doctor(cluster, nodes, load("upstreams.lock.json"))
            return [f for f in findings if "sizing.max_num_seqs" in f]

        to_32 = recipe.capture_sizes(32)
        # #434: six streams' maximum (36) over a ladder that stops at 32.
        refused = profile(6, 36, to_32)
        errors, warnings = recipe.capture_problems(refused)
        self.assertEqual(warnings, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("sizing.max_num_seqs (6) x (1 + 5 drafts) = 36 rows", errors[0])
        self.assertIn("end at 32 and --max-cudagraph-capture-size is 36", errors[0])
        (finding,) = capture_findings(refused)
        self.assertNotIsInstance(finding, spark.Warn)
        # Ten streams (60 rows) under eight streams' ladder boot, and their full decode steps
        # run without a graph.
        stale = profile(10, 48, recipe.capture_sizes(48))
        errors, warnings = recipe.capture_problems(stale)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("up to 48 rows, below sizing.max_num_seqs (10) x (1 + 5 drafts) = 60 rows", warnings[0])
        (finding,) = capture_findings(stale)
        self.assertIsInstance(finding, spark.Warn)
        # Six streams under eight streams' ladder are consistent: vLLM boots and every step fits.
        self.assertEqual(recipe.capture_problems(profile(6, 48, recipe.capture_sizes(48))), ([], []))
        # The recipe's own rendering is consistent.
        self.assertEqual(capture_findings(base), [])

    def test_each_shape_pins_its_prefill_budget(self):
        """Each shape's long-prefill budget is its own sweep's answer (a TP3 rank holds a
        third of the model, so its headroom differs): changing one must not move another.
        The threshold is the budget less streams x (1 + drafts)."""
        expected = {"tp4-ring": ("16384", "16288"), "tp3-triangle": ("4096", "4048"),
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
