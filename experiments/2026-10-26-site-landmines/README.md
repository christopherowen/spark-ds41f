# Site landmines: the cache mount and the image cap (2026-10-26)

Two things that break a site the recipe was not written on. The owner saw both in a report on
2026-10-25: the hard-coded cache mount path, and the cap of 4 images a request.

## 1. The cache mount follows the checkout (done)

`config/cluster.json` mounted `{home}/projects/spark-ds41f/cache` at `/cache`. `deployment.path`
named the checkout separately, and `bin/spark` filled only `{home}`. So a site that cloned
somewhere else and set `deployment.path` still mounted our cache path, and the start refused with
`rw mount source is unavailable`.

The fix:
- A `{checkout}` placeholder is filled with `deployment.path` everywhere `{home}` is, through
  one helper (`host_path` in `bin/spark`): the binds, the mount test, the checkout's status
  exclusions and the weight-fragmentation root.
- `scripts/lab.py`'s fence check and overlay manifest resolve an arm's mounts the same way. With
  `{home}` alone, the recipe's cache mount would have raised `KeyError` at a lab run's first boot.
- The recipe's cache mount is `{checkout}/cache`. `deployment.path` may name only `{home}`.

Our resolved commands are unchanged. On every shape (`tp4-ring`, `tp3-triangle`,
`tp3-triangle-4k`) and every node, these are byte-identical before and after:
- the rendered `docker run` command;
- the binds `doctor --live` compares;
- the mount test and the checkout's status command.

`bin/spark --shape <shape> doctor` prints the same report before and after (`configuration OK`).
The tests check that a site with another `deployment.path` mounts its own cache, and that ours
resolves to the production binds. [docs/setup.md](../../docs/setup.md#4-site-configuration)
describes `deployment.path` and the two placeholders. The frozen 1022 recipe and the experiment
records keep their `{home}` paths.

## 2. The per-request image cap

### What it does

`--limit-mm-per-prompt {"image":4}` makes vLLM refuse a request with more than four images
(HTTP 400). A chat client resends its history, so a conversation's fifth image fails every later
turn. It was chosen with the vision tower on 2026-09-25
([vision](../2026-09-25-vision/README.md)), with no budget reason recorded. An external TP3 site
served 13 uncapped
([#30](https://github.com/christopherowen/spark-ds41f/issues/30)).

What an image costs, stage by stage, on 1017's processor (`vision_config`: patch 14,
downsample 3, at most 1,024 tokens):
- **Tokens and KV.** A larger source is shrunk to fit the token budget: a square tops out at
  1,302 px, 994 tokens and 8,649 ViT patches. Every source here, from that square to 178.9 MP,
  cost 994-1,001 tokens. 32 such images are about 32K tokens from the fixed pool (TP3 holds 4.0-4.8
  million tokens); eight such streams take about 6% of it. The pool is preallocated, so KV use
  does not move MemAvailable.
- **Encoder.** vLLM encodes a request's images within a per-step budget (the batched-token
  budget, 4-8 images a step on the triangle). So the tower's working memory does not grow with
  the count. Startup profiling is bounded the same way: min(budget / 1,024, 8 streams x cap) is
  already the encoder bound at a cap of 4, so a larger cap profiles the same inputs.
- **Patches.** 9.7 MiB of BF16 patches per token-max image, whatever its source size. The
  engine core holds them until the image is encoded, and every rank's worker receives them (the
  tower runs data-parallel across ranks). This is the count-driven host cost, measured on the
  cluster below on every node.
- **The API server's decode (dgx1).** vLLM decodes every image of every request on one pool of
  eight threads (`VLLM_MEDIA_LOADING_THREAD_COUNT`). DS4.1's processor then shrinks each
  request's images, one request at a time (the renderer has one multimodal worker). The decoded
  size follows the source's pixels, not the tokens. vLLM accepts up to `VLLM_MAX_IMAGE_PIXELS`,
  178,956,970 pixels by default, and sets no request-size limit. The parsed request keeps
  every image's base64 text for as long as the request streams.

### The API server's image path, measured on dgx4 (done)

[bundles/image-host-memory](bundles/image-host-memory) ran
[image_host_memory.py](image_host_memory.py) through `scripts/lab.py kernel-local` on dgx4.
Details:
- 2026-10-25, 22:12-22:16 CET; image `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b5`);
  CPU only; the recipe's glibc settings.
- It runs vLLM's own decoder (`ImageMediaIO.load_base64`, eight threads) and DS4.1's processor
  for K requests of N images.
- Each case is a fresh process: peak and held RSS above its imports.
- Everything stays referenced to the end of a case. That is the bound for a burst of requests
  queued behind the one preprocessing worker.
- Record: [runs/dgx4-image-host-memory](runs/dgx4-image-host-memory) (31 cases, none skipped,
  verdict pass).

Per image in flight (MiB). The one-image cases include about 80 MiB of first use that a serving
API server has already paid; the slopes are from the larger cases.

| Source | Upload (base64) | One image: peak / held | Held per extra image | One request of 4: peak |
|---|---|---|---|---|
| 1,302 px square, JPEG (the token maximum) | 0.62 | 101 / 101 | 18.8 (8 x 16 to 8 x 32) | 166 |
| 12 MP photo (4,032 x 3,024), JPEG | 4.07 | 148 / 148 | 61-74 | 360 |
| 48 MP photo (8,064 x 6,048), JPEG | 15.6 | 435 / 308 | (one image measured) | |
| 48 MP, PNG with alpha | 24.8 | 593 / 323 | | |
| 178.9 MP (vLLM's limit), JPEG | 56.1 | 1,537 / 875 | 789 (1 to 4) | 5,358 |
| 178.9 MP, PNG with alpha | 57.0 | 2,144 / 876 | | 8,592 |

Several requests at once:
- 8 requests of 32 token-max images: 4.7 GiB.
- 8 requests of 8 12 MP photos: 4.1 GiB.
- 4 requests of 4 178.9 MP PNGs: 18.2 GiB peak.

An alpha PNG's peak is the largest: the RGBA decode, its RGB copy and the alpha band are alive
together, in up to eight threads at once.

### The landmine under the cap: source pixels

dgx1's margin above its 3 GiB steady guard, on spark-ds41f-1025 at TP3's 8,192 budget
([its TP3 qualification](../2026-10-25-spark-ds41f-candidate/README.md)):
- 4.4 GiB once serving (7.36 GiB MemAvailable);
- 3.7 GiB at its lowest through three cold 1M prefills (6.67 GiB).

The cap of 4 bounds tokens, not the API server's memory. Against that margin:
- One request of four 178.9 MP images peaks at 5.2 GiB (JPEG) or 8.4 GiB (alpha PNG). The
  upload is about 225 MiB.
- Two alpha PNGs at vLLM's limit in one request are about 4.2 GiB.
- Four requests of four 48 MP photos are about 4-6 GiB (scaled from one image).

Each of these takes dgx1 to or through its guard, and the guard stops the whole service. The
recipe allows all of them today. A count measured with token-max images says nothing about them,
so a count only means something with a pixel bound beside it.

Proposed, for the owner (not applied):
- **`VLLM_MAX_IMAGE_PIXELS` in the recipe's environment.** vLLM checks it before decoding and
  refuses a larger image with HTTP 400 ("Image dimensions ... exceed the maximum of ... pixels").
  The model never sees more than about 1.7 MP (sources are shrunk to 994-1,001 tokens), so the
  bound changes no answer. At 16,777,216 (16 MP; 4,096 x 4,096):
  - per image, about 75-90 MiB held and about 200 MiB peak (scaled from the 12 MP and 178.9 MP
    cases);
  - accepted: 12 MP photos, 4K and 5K screenshots;
  - refused: 48 MP full-resolution photos, which a client must downscale.
- **`VLLM_MEDIA_LOADING_THREAD_COUNT`** (eight decode threads) bounds how many peaks overlap.
  Fewer threads lower the transient and add queueing for bursts. It is secondary to the pixel
  bound.

### The cluster measurement (queued, not run)

[image-cap-tp3.json](image-cap-tp3.json) boots
[image-cap-32-tp3.json](image-cap-32-tp3.json): the recipe at `tp3-triangle` with
`{"image":32}` and nothing else changed: spark-ds41f-1025 at the 8,192 budget.
[make_arms.py](make_arms.py) writes the arm, and `tests/test_recipe.py` fails if the recipe's
triangle moves away from it.

The spec's extras:
1. startup lines;
2. a memory snapshot;
3. [image_cap.py](image_cap.py);
4. the image digest, as a check that images still answer;
5. a second memory snapshot.

`image_cap.py` runs on the head, as every extra does:
- **Images.** A step is STREAMS concurrent requests of COUNT distinct token-max images, never
  cached. Each is a PNG of about 0.54 MiB, a 1,302 px photo's JPEG size.
- **Steps.** 4, 8, 16 and 32 images at 1, 4 and 8 streams (TP3 serves eight), each twice, in
  order of images in flight, after a one-image warm-up.
- **Per step, it records:**
  - every node's MemAvailable low (dgx1 every 0.2 s, the others every second);
  - the head container's API-server, engine-core and worker RSS;
  - the KV usage peak from `/metrics`;
  - each request's status, prompt tokens, upload time and TTFT;
  - its own RSS, which dgx1's MemAvailable includes (at most about 200 MiB, at 32 x 8).
- **Gate.** A step is not sent if the largest drop per image so far, times 1.25, predicts a node
  below 4.5 GiB. A node below 4.5 GiB during a step cancels that step's requests and ends the
  sweep. 4.5 GiB is the 3 GiB guard plus the 1.5 GiB margin the cap is chosen for. The guards
  are not touched.
- **Validation.** Last, 8 12 MP photos (3.84 MiB PNGs) at one stream, gated with their decode
  cost added. The API server's extra RSS over the token-max step should match the dgx4 table: it
  checks the bundle against the real server.

Before it runs:
- merge this branch, since the lab syncs the nodes to `origin/main`;
- regenerate the arm if the recipe changed since;
- pass the `--production-config` of the TP3 release then serving (`config/cluster.json` since
  spark-ds41f-1025 serves the triangle);
- check with the coordinating session (local_0b4b0a6b), as for anything on dgx1-3.

About 40 minutes with the boot.

**Decision rule.** The cap becomes the highest count whose worst measured low on dgx1, at the
most streams it ran, stays at or above 4.5 GiB with token-max images. That count holds only for
sources whose decode cost the measured drop covers. The recipe change therefore lands with the
pixel bound, rechecked against the dgx4 table:
- count x 8 streams x (held per image at the bound - 18.8 MiB) + eight decode peaks must fit in
  what the sweep left above 4.5 GiB;
- otherwise the count comes down.

The recipe and its docs change after the measurement; nothing in serving changes here.

## Files

- [image_host_memory.py](image_host_memory.py), [bundles/image-host-memory](bundles/image-host-memory):
  the dgx4 bundle. Its record is in [runs/dgx4-image-host-memory](runs/dgx4-image-host-memory).
- [image_cap.py](image_cap.py): the cluster client (a lab extras script; stdlib only).
- [make_arms.py](make_arms.py), [image-cap-32-tp3.json](image-cap-32-tp3.json): the arm.
- [image-cap-tp3.json](image-cap-tp3.json): the lab spec.
