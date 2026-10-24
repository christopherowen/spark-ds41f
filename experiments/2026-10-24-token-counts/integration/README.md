# Integration after the runner-input port

The standalone series deliberately measures only this kernel on candidate 4's
source. The concurrent runner-input port changes the context around the last
Triton warmup and shortens `register_input_batch_warmups` to a `dead_rows` argument.
A straight application of the standalone patch therefore conflicts in that file.

[after-runner-port.patch](after-runner-port.patch) is the same port adapted to
runner commit `1bc9f13f620c12ea07b5fd0d8542d1758fb81825`. It preserves that owner's
API and updates the declaration test to call it. The remaining Triton warmup
imports become unused and are removed. Prepared and checked in a separate
throwaway checkout; the concurrent port's checkout was not modified.

The kernel and benchmark are byte-identical to the GPU-tested candidate. The
public count wrapper and unchanged Triton reference have identical ASTs. Ruff
and whitespace pass. This does not qualify the combined runner source on GPU;
its full release and serving validation remain required.

Use this patch **instead of** the standalone patch when integrating after that
runner port. It is not included in the standalone full series.
[Source identities](source.json) pin the adjustment and its base.
