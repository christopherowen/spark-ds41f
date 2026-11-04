# Additional memory-ownership audit

Read-only source inspection of candidate-39 plus request-submission cleanup
(`0cff1109940f15d08f2e893ae06e0a527fde438a`), with candidate-38 comparisons
for the changed client module. Only the utility-call correction is tested in
this experiment. No serving process was instrumented or restarted.

## Demonstrated additional leak

`vllm/v1/engine/core_client.py`: both utility clients register their future
before encoding/sending but previously removed it only on reply. Every failed
send leaves an entry when no reply can arrive. Cancellation can leave a pending
or cancelled future until an eventual reply, with no bound if it never arrives.
The Spaceport control's repeated-failure tests leave 64 entries per client;
the corrected clients leave none. The same module bytes are present in
candidate-38 and the inspected candidate-39 recipe.

These entries are much smaller than full prompt states. The test establishes
retention and cleanup, not a GiB/hour rate or relevance to #884's traffic.
Healthy utility calls already removed their entries; this is an exceptional
path leak, not growth on every successful inference request.

## Other ownership paths inspected

| Component | Source observation | Assessment within this audit |
| --- | --- | --- |
| Frontend prompt/output state | `OutputProcessor._finish_request()` and `abort_requests()` remove request and external-ID entries; parallel sampling removes its parent. The preceding experiment corrects the submission exception gap. | No additional missing normal-completion removal found. |
| Output collector | One pending output, with delta merging rather than a queue per token. | Slow-consumer output can grow during a live request; not an accumulating completed-request collection. |
| GPU request slots and CopyDrafter | Request maps remove finished IDs; token/match/draft tensors use configured capacities; staged row lists clear after writes. | No new unbounded collection identified. Fixed capacities are not a leak-rate measurement. |
| DS4.1 model host state | `DeepseekV4ModelState.remove_request()` removes prefill token arrays, CED prefix/logprob entries and steering IDs. Runner removal calls it before releasing the request slot. | Prompt arrays have a matching removal hook. |
| Prompt embeddings and prompt logprobs | Their dictionaries have removal methods called by the runner; completed chunked logprob lists also clear. | No missing request removal found in these paths. |
| Encoder outputs | Request features are removed on request release; cached encoder outputs are removed by scheduler-directed hash frees. Multimodal frontend caches use size-bounded LRU storage. | Cache occupancy can warm up; inspection does not establish its live occupancy or allocator behavior. |
| Engram native reader | Early-ID arrays are freed by the worker after queued work, or by submit/prefetch when not queued. Queue capacity is 64. Reader/planner persistent buffers have destruction paths; the row cache has a configured fixed size. | No unmatched per-step allocation found in the reviewed submit/prefetch/worker paths. No native heap trace was taken. |
| Serialization | `MsgpackEncoder.encode()` and `encode_into()` reset temporary auxiliary buffers in `finally`. | No accumulating serializer buffer list found. |
| Parser configuration | V4.1 configuration cache keys are the thinking boolean; streaming parser state is per request. | This configuration cache does not add an entry per request. |
| Structured output | Grammar compiler cache is explicitly byte-limited; reasoner instance belongs to the request. | Cache warm-up remains distinct from an unbounded leak. |

These are source observations, not blanket clearance of every error path,
native allocation or third-party cache.

## Conditional retention outside the promoted recipe

- The Responses API's `response_store`, `msg_store` and `event_store` are
  unbounded when `VLLM_ENABLE_RESPONSES_API_STORE=1`. The implementation warns
  about this. The setting defaults to false and is absent from the promoted
  and reviewed candidate recipes; background requests require enabled storage.
  Do not silently evict API history or change this feature's policy in a memory
  correction unrelated to that API. This path can matter on an externally
  customized deployment, but it has not been tied to #884.
- The data-parallel load-balancing client records routing before submission
  and removes it on a finished-request report. A send that never reaches an
  engine has no such report. This is a further exception-path ownership concern
  to qualify for data-parallel serving; the TP3/TP4 single-engine recipes do
  not instantiate that client. It is not exercised or corrected here.

## Remaining field attribution

Neither the request-submission correction nor this utility correction identifies
which process grew during #884. The report has host MemAvailable but no growing
process's RSS/allocation ownership and no count of failed/cancelled submissions
or utility calls. The defects are shared host code, so they do not by themselves
explain a candidate-38-only regression. Keep the field issue open. No context,
KV capacity, memory guard or allocator setting was reduced to pass these tests.
