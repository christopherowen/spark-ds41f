# Why the asynchronous Engram rows never arrive

Owner, 2026-10-15: "we should solve the reason the engram is timing out. it should
return instantly."

## What the traced repro showed

The Poisson first-request crash ([2026-10-14-poisson-crash](../2026-10-14-poisson-crash))
reproduced on 3 of 8 traced boots (t1, t4, t6), identically:

- Every rank stages Engram sequence 27 (the first decode step after the first request's
  prefill). Ranks 0-2 never finish reading it ("rows read" is missing for both tables);
  rank 3 reads it at once. The successful boot (t2) read all 631 sequences on every rank.
- Ranks 0-2's GPUs stop at the first decode step's Engram rows wait. Rank 3 passes it and
  enters the Engram rows all-reduce (sparknet sequence 4855, 49,152 bytes, the same op
  every rank runs in t2). No peer arrives; after about 5.5 s the one-shot kernel gives
  up and returns garbage, and the router's NaN check fires on rank 3. The NaN is a
  symptom.
- The reader's 20 s deadline never logged and the gate was never released: the reader
  thread never got back to Python, or the device work it waits for never ran.
- Rank 3 reached that step about 100 ms after the others in every failing boot (the
  others waited 75-113 ms for it at sequence 4852; 3-21 ms in t2).
- Ruled out: the DSpark cost tables (pinned, broadcast from rank 0), the
  boundary-replay and speculative-depth branches (scheduler output), lazy JIT compiles.

The read itself is 96 rows. It should return at once; something stops the handoff.

## What this run shows ([stall.json](stall.json))

Arms use r6k's image with an instrumented `engram.py` overlay
([overlay/watch](overlay/watch)), debug-only:

- reader phase timings (`ids wait`, `ids ready after`, `read`) with monotonic timestamps,
  and an immediate log line when the reader's wait for its row IDs times out (today the
  error waits for the next step's join, which never comes);
- a stall watcher (`SPARK3_ENGRAM_STALL_WATCH_S=3`): while a staged sequence stays unread
  or its rows flag stays 0, it logs every table's gate, flag, `ids_ready` and
  `cache_done` states and every Python thread's stack;
- [stall_probe.py](stall_probe.py): one request per fresh boot; if it has not finished
  in 45 s, `sudo gdb -batch` backtraces of every thread of every vLLM engine and worker
  process on every node, and the serving logs, under `results/private/engram-stall/`.

Boots: `delay.json` (d1, d2: the reader sleeps 200 ms before each read, so it is always
later than the forward) and `watch.json` (w1-w4: production timing). If the hang needs
the forward to reach its Engram wait before the reader releases the rows, the delay arm
hangs on every boot.
