"""Which shared-expert schedule does the serving FFN sublayer take? (eager, traced)"""
import sys
import time

sys.path.insert(0, "/h")
import torch

import ffnlayer as F

I = int(sys.argv[1]) if len(sys.argv) > 1 else 576
t0 = time.time()
vc = F.make_config(I)
print("config", time.time() - t0, "quant", type(vc.quant_config).__name__,
      "v2", vc.use_v2_model_runner, "mnbt", vc.scheduler_config.max_num_batched_tokens)
F.init_dist(vc)
layers = F.build_layers(vc, 1, verbose=True)
print("built", time.time() - t0)
layer = layers[0]
runner = layer.ffn.experts
print("runner", type(runner).__module__, type(runner).__name__)
shared = runner._shared_experts
print("shared", type(shared).__name__, "stream", shared._stream)
qm = runner._quant_method
print("quant_method", type(qm).__name__, "modular_experts", type(qm.modular_experts).__name__)
print("router", type(runner.router).__name__, "routes_hidden_states", getattr(runner.router, "routes_hidden_states", None))
print("routed_scaling_factor (runner)", runner.routed_scaling_factor, "transform", runner.routed_output_transform)
print("is_monolithic", runner.routed_experts.quant_method.is_monolithic)
print("gate on runner", runner.gate)
print("shared act", type(layer.ffn.shared_experts.act_fn).__name__,
      "gate_up mxfp8_input", getattr(layer.ffn.shared_experts.gate_up_proj, "tilelang_mxfp8_input", None),
      "down mxfp8_input", getattr(layer.ffn.shared_experts.down_proj, "tilelang_mxfp8_input", None))

from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.models.deepseek_v4_1.tilelang import router as router_mod

events = []
orig_async = se_mod.SharedExperts.maybe_forward_async
orig_sync = se_mod.SharedExperts.maybe_sync_shared_experts_stream
orig_cr = router_mod.TileLangGateRouter._compute_routing


def traced_async(self, *a, **k):
    events.append(("shared.maybe_forward_async", k.get("input_ready_recorded", False)))
    return orig_async(self, *a, **k)


def traced_sync(self, *a, **k):
    events.append(("shared.maybe_sync_shared_experts_stream",))
    return orig_sync(self, *a, **k)


def traced_cr(self, *a, **k):
    events.append(("router._compute_routing", "post_gate_hook" if getattr(self, "_post_gate_hook", None) else "no hook"))
    return orig_cr(self, *a, **k)


se_mod.SharedExperts.maybe_forward_async = traced_async
se_mod.SharedExperts.maybe_sync_shared_experts_stream = traced_sync
router_mod.TileLangGateRouter._compute_routing = traced_cr
if hasattr(se_mod.SharedExperts, "record_input_ready"):
    orig_rir = se_mod.SharedExperts.record_input_ready

    def traced_rir(self, *a, **k):
        events.append(("shared.record_input_ready",))
        return orig_rir(self, *a, **k)

    se_mod.SharedExperts.record_input_ready = traced_rir

for M in (1, 6, 48):
    inp = F.FFNInputs(layer, M)
    events.clear()
    with F.forward_context(vc, inp):
        x0 = torch.randn(M, 5120, device="cuda", dtype=torch.bfloat16)
        print("M", M, "workspace_size", shared.workspace_size(x0), "order",
              shared._determine_shared_experts_order(x0).name,
              "fold", type(runner._shared_output_fold(x0)).__name__)
        out = F.ffn_sublayer(layer, inp)
    torch.cuda.synchronize()
    print("M", M, "events", events)
    print("M", M, "out", out[0].shape, out[0].dtype, "finite", bool(torch.isfinite(out[0].float()).all()),
          "absmean", float(out[0].float().abs().mean()))
print("done", time.time() - t0)
