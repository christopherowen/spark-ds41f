
#### tp4-ring: production

| Mix | Users wait on | TTFT p50 | TTFT p99 | Their ITL p99 | Long prompt TTFT p50 | Output tok/s | Recomputed tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poisson | chat | 0.27 s | 3.54 s | 0.14 s | - | 176.7 | 0 |
| stall | probe | 12.91 s | 18.91 s | 2.42 s (stall step) | 20.8 s | 135.2 | 0 |
| stall-1m | probe | 132.14 s | 202.14 s | 3.19 s (stall step) | 212.0 s | 32.3 | 0 |
| chat-docs | chat | 0.44 s | 84.18 s | 1.12 s | 40.3 s | 118.8 | 0 |
| agents-c8 | agent | 1.13 s | 8.88 s | 0.96 s | - | 138.7 | 0 |
| agents-c16 | agent | 2.54 s | 30.21 s | 1.54 s | - | 159.8 | 49 |
| agents-c24 | agent | 20.83 s | 63.23 s | 1.77 s | - | 152.0 | 244 |
| agents-c8+chat | chat | 0.80 s | 23.25 s | 1.21 s | - | 168.3 | 1 |
| agents-c8+chat-heavy | chat | 29.99 s | 79.06 s | 0.93 s | - | 204.1 | 0 |
| soak | chat | 0.34 s | 14.46 s | 3.95 s | 13.9 s | 68.1 | 0 |

#### tp4-ring: change against production (user TTFT p50 / p99, output, long-prompt TTFT)

| Mix | prefill yield | slots | prefill yield + slots | chunks 8K beside decodes | chunks 4K beside decodes | graphed decode rows* | held first token off* | pool 2x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poisson | +0% / +0%, out +0% | +0% / -87%, out +0% | +0% / -87%, out +0% | +0% / +0%, out +0% | +0% / +0%, out +0% | +0% / +0%, out +0% | -30% / -3%, out +0% | +0% / +0%, out +0% |
| stall | -72% / -69%, out +4%, long +11% | +1% / +0%, out +0%, long +0% | -72% / -69%, out +4%, long +11% | -71% / -61%, out +3%, long +12% | -81% / -82%, out +6%, long +35% | -4% / -3%, out +0%, long -2% | -1% / -1%, out +0%, long +0% | +0% / +0%, out +0%, long +0% |
| stall-1m | -95% / -95%, out +4%, long +1% | +0% / +0%, out +0%, long +0% | -95% / -95%, out +4%, long +1% | -96% / -97%, out +17%, long +9% | -98% / -98%, out +44%, long +26% | -3% / -2%, out +0%, long -2% | -0% / -0%, out +0%, long +0% | +0% / +0%, out +0%, long +0% |
| chat-docs | +1% / -13%, out -0%, long +1% | -3% / -13%, out -0%, long -11% | -3% / -50%, out -0%, long -6% | +9% / +3%, out -0%, long +13% | +33% / +22%, out -0%, long +35% | -5% / -3%, out -0%, long -2% | -18% / -0%, out -0%, long -2% | +0% / +0%, out +0%, long +0% |
| agents-c8 | -0% / -32%, out -0% | -0% / +7%, out +0% | -1% / -31%, out -0% | -2% / +11%, out -0% | -0% / -22%, out -2% | -6% / +11%, out +2% | -11% / -14%, out +0% | -0% / +0%, out +0% |
| agents-c16 | -6% / -8%, out -0% | -47% / -61%, out +3% | -48% / -78%, out +2% | -7% / -3%, out -0% | -1% / +2%, out -4% | -17% / -9%, out +4% | -12% / -0%, out +0% | -5% / -10%, out +1% |
| agents-c24 | +5% / -1%, out -0% | -83% / -22%, out +5% | -83% / -35%, out +4% | +5% / +1%, out +1% | +8% / -3%, out -3% | -1% / +3%, out +3% | -1% / -1%, out +0% | +0% / -17%, out +4% |
| agents-c8+chat | -2% / -3%, out -0% | -25% / -52%, out +1% | -21% / -75%, out +1% | +3% / -7%, out -0% | +10% / +6%, out -2% | -9% / -14%, out +2% | -21% / -0%, out -0% | +0% / +0%, out -0% |
| agents-c8+chat-heavy | +2% / +6%, out +0% | -77% / -49%, out +7% | -82% / -36%, out +7% | +1% / -2%, out -0% | +6% / +10%, out -2% | -6% / +12%, out +1% | -0% / -0%, out +0% | +0% / +0%, out +0% |
| soak | +0% / -34%, out +0%, long -8% | +0% / -1%, out +0%, long -1% | +0% / -34%, out +0%, long -10% | +0% / -36%, out -1%, long +17% | +0% / -40%, out -2%, long +46% | +0% / -3%, out +1%, long -2% | -30% / -22%, out -0%, long -13% | +0% / +0%, out +0%, long +0% |

#### tp3-triangle: production

| Mix | Users wait on | TTFT p50 | TTFT p99 | Their ITL p99 | Long prompt TTFT p50 | Output tok/s | Recomputed tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poisson | chat | 0.26 s | 7.97 s | 0.14 s | - | 105.2 | 0 |
| stall | probe | 16.46 s | 22.46 s | 1.45 s (stall step) | 24.4 s | 71.2 | 0 |
| chat-docs | chat | 0.32 s | 31.65 s | 0.14 s | 17.6 s | 54.9 | 0 |
| agents-c4 | agent | 1.00 s | 10.42 s | 0.64 s | - | 87.4 | 151 |
| agents-c2+chat | chat | 0.38 s | 8.51 s | 0.45 s | - | 82.6 | 3 |
| soak | chat | 0.30 s | 7.79 s | 2.48 s | 12.0 s | 52.8 | 0 |

#### tp3-triangle: change against production (user TTFT p50 / p99, output, long-prompt TTFT)

| Mix | prefill yield | slots | prefill yield + slots | chunks 4K beside decodes | retention 2,048 | pool +1.3 GiB | held first token off* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| poisson | +0% / +0%, out +0% | -1% / -84%, out +0% | -1% / -84%, out +0% | +0% / +0%, out +0% | +0% / +0%, out +0% | +0% / +0%, out +0% | -31% / -1%, out +0% |
| stall | -83% / -82%, out +4%, long +8% | +1% / +1%, out +0%, long +1% | -83% / -83%, out +4%, long +8% | -85% / -80%, out +5%, long +17% | +0% / +0%, out +0%, long +0% | +0% / +0%, out +0%, long +0% | -1% / -0%, out +0%, long +0% |
| chat-docs | -0% / -54%, out -0%, long +5% | -0% / -16%, out -0%, long -2% | -0% / -87%, out -0%, long +0% | +1% / -12%, out +0%, long +7% | +0% / +0%, out +0%, long +0% | +0% / +0%, out +0%, long +0% | -20% / -1%, out +0%, long -3% |
| agents-c4 | -0% / -45%, out +0% | -1% / -0%, out +0% | +0% / -42%, out +0% | -2% / -27%, out -1% | +2% / -22%, out +2% | +0% / -22%, out +2% | -9% / -9%, out +0% |
| agents-c2+chat | -0% / -47%, out -0% | +1% / -11%, out -0% | +1% / -56%, out -1% | -0% / -26%, out -1% | +1% / -4%, out -0% | +1% / -2%, out +0% | -20% / -6%, out -0% |
| soak | +3% / -19%, out -0%, long -1% | +2% / -5%, out +0%, long -0% | +3% / -18%, out -0%, long -1% | +3% / -40%, out -2%, long +8% | +0% / +0%, out +0%, long +0% | +0% / +0%, out +0%, long +0% | -30% / -31%, out -0%, long -9% |
