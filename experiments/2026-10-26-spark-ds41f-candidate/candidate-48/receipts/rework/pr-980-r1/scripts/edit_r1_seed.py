"""edit_r1_seed.py <vllm-src>: key SamplingStates._seed_rng to the configured seed.

states.py: SamplingStates(max_num_reqs, vocab_size, seed=None); the unseeded-request
generator is np.random.default_rng of the configured seed, vLLM's default 0 when none.
sampler.py: Sampler(..., seed=None) hands it to SamplingStates.
model_runner.py: the sampler kwargs carry ModelConfig.seed (what the worker passes to
set_random_seed).
tests: different configured seeds give different unseeded seeds; ranks with one seed
agree; no seed equals seed 0; the Sampler forwards its seed."""
import sys

root = sys.argv[1]


def edit(rel, pairs):
    p = f"{root}/{rel}"
    s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    open(p, "w").write(s)


edit("vllm/v1/worker/gpu/sample/states.py", [
    (
        "_NP_INT64_MAX = np.iinfo(np.int64).max\n",
        "_NP_INT64_MAX = np.iinfo(np.int64).max\n"
        "\n"
        "\n"
        "def unseeded_request_seed_rng(seed: int | None) -> np.random.Generator:\n"
        "    \"\"\"The generator of unseeded requests' seeds for configured seed ``seed``\n"
        "    (ModelConfig.seed, the value the worker passes to set_random_seed).\n"
        "\n"
        "    It depends on nothing rank-local, so every TP rank draws the same seed for\n"
        "    each unseeded request, and a different ``--seed`` gives a different stream.\n"
        "    ``None`` (no configured seed) is vLLM's default seed, 0.\n"
        "    \"\"\"\n"
        "    return np.random.default_rng(0 if seed is None else int(seed) % (1 << 64))\n",
    ),
    (
        "    def __init__(self, max_num_reqs: int, vocab_size: int):\n",
        "    def __init__(self, max_num_reqs: int, vocab_size: int, seed: int | None = None):\n",
    ),
    (
        "        # Dedicated lockstep generator so unseeded requests draw identical\n"
        "        # per-request seeds across TP ranks regardless of rank-local callers\n"
        "        # touching the process-global NumPy RNG.\n"
        "        self._seed_rng = np.random.default_rng(0)\n",
        "        # Dedicated lockstep generator so unseeded requests draw identical\n"
        "        # per-request seeds across TP ranks regardless of rank-local callers\n"
        "        # touching the process-global NumPy RNG, keyed by the configured seed.\n"
        "        self._seed_rng = unseeded_request_seed_rng(seed)\n",
    ),
    (
        "            if seed_rng is None:\n"
        "                seed_rng = np.random.default_rng(0)\n",
        "            if seed_rng is None:\n"
        "                seed_rng = unseeded_request_seed_rng(None)\n",
    ),
])

edit("vllm/v1/worker/gpu/sample/sampler.py", [
    (
        "        return_sampling_mask: bool = False,\n"
        "    ):\n"
        "        self.logprobs_mode = logprobs_mode\n",
        "        return_sampling_mask: bool = False,\n"
        "        seed: int | None = None,\n"
        "    ):\n"
        "        self.logprobs_mode = logprobs_mode\n",
    ),
    (
        "        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)\n",
        "        # Unseeded requests' seeds follow the configured seed (ModelConfig.seed).\n"
        "        self.sampling_states = SamplingStates(max_num_reqs, vocab_size, seed=seed)\n",
    ),
])

edit("vllm/v1/worker/gpu/model_runner.py", [
    (
        "                \"return_sampling_mask\": self.model_config.return_sampling_mask,\n"
        "            }\n",
        "                \"return_sampling_mask\": self.model_config.return_sampling_mask,\n"
        "                # The seed the worker passes to set_random_seed: unseeded\n"
        "                # requests' seeds follow it, identically on every TP rank.\n"
        "                \"seed\": self.model_config.seed,\n"
        "            }\n",
    ),
])

test_path = f"{root}/tests/v1/spec_decode/test_adaptive_verification.py"
with open(test_path, "a") as f:
    f.write('''

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SamplingStates keeps UVA buffers"
)
def test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks():
    """Unseeded requests' seeds come from a stream keyed by the configured seed
    (ModelConfig.seed): ranks built with the same seed assign the same seeds however
    each perturbs the global NumPy RNG, a different seed gives different seeds, and no
    configured seed is vLLM's default seed 0."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    def unseeded_seeds(seed, perturb):
        np.random.seed(perturb)
        _ = np.random.randint(0, 1000, size=perturb % 41 + 1)
        states = SamplingStates(8, 1024, seed=seed)
        for slot in range(8):
            states.add_request(slot, SamplingParams(temperature=0.8, seed=None))
            _ = np.random.random(perturb % 7 + 1)
        assert not states.seeds_set.any()
        return states.seeds.np.copy()

    rank0 = unseeded_seeds(1234, perturb=1)
    rank1 = unseeded_seeds(1234, perturb=99999)
    np.testing.assert_array_equal(rank0, rank1)
    assert len(set(rank0.tolist())) == 8

    other_seed = unseeded_seeds(4321, perturb=1)
    assert set(rank0.tolist()).isdisjoint(other_seed.tolist())
    np.testing.assert_array_equal(other_seed, unseeded_seeds(4321, perturb=12345))

    np.testing.assert_array_equal(unseeded_seeds(None, 5), unseeded_seeds(0, 7))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Sampler keeps UVA buffers"
)
def test_sampler_hands_configured_seed_to_sampling_states():
    """The model runner's sampler kwargs carry ModelConfig.seed; the Sampler must
    key its unseeded-request seeds by it."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")

    def first_unseeded_seed(seed):
        req_states = RequestState(
            max_num_reqs=4,
            max_model_len=64,
            max_num_batched_tokens=16,
            num_speculative_steps=1,
            vocab_size=128,
            device=device,
        )
        sampler = Sampler(
            max_num_reqs=4,
            vocab_size=128,
            device=device,
            req_states=req_states,
            seed=seed,
        )
        sampler.add_request(0, 1, SamplingParams(temperature=0.8, seed=None))
        return int(sampler.sampling_states.seeds.np[0])

    assert first_unseeded_seed(7) == first_unseeded_seed(7)
    assert first_unseeded_seed(7) != first_unseeded_seed(8)
''')
print("edited")
