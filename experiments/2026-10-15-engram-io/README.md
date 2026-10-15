# Engram rows from a native reader, gathered in the forward

Owner, 2026-10-15: solve why Engram timed out ("it should return instantly"), and end
with "a cleaner architecture that is quicker and easier to reason about".

[2026-10-15-engram-stall](../2026-10-15-engram-stall) found the cause: the Python
reader thread needed the GIL to release a step's rows, and the main thread held the GIL
in a lazy `cuModuleLoad` (the torch.compile'd draft-budget helper's first call) while
waiting for the GPU, which was waiting for those rows.

## The change (vLLM branch `engram-io`)

1. `4435be337`: the draft-budget selection runs eagerly with its kernels loaded at
   startup; the JIT monitor reports torch.compile compilations during inference.
2. `69630052d`: per table and step, the model's stream copies the row IDs to pinned host
   memory and writes the step's sequence number into a control block in mapped host
   memory; a native worker (`engram_rows/_reader.c`, B12X's PLE planner and io_uring
   reads) reads the rows into mapped buffers and publishes the step, never taking the
   GIL or calling CUDA; the Engram layer's forward gathers with one TileLang kernel that
   waits for the step's rows (bounded; a timeout records the step and stops the GPU) and
   dequantizes them bit for bit as B12X's lookup did. The Python reader thread, side
   stream, CUDA events, flags, gate, wait op, synchronous path and disk lookup plan are
   gone.

A CPU-only run of the reader (Python writing the GPU's word) read 50 random steps
correctly, ~70 us per 96-row step, named a missing source and an ID timeout.

## This run ([engram-io.json](engram-io.json))

On r6k's image with the branch's files mounted
([bundles/tests/overlay](bundles/tests/overlay); `model_runner.py` is r6k's plus the
startup call):

- kernel-lab tests (dgx4, `io_uring` bundles): the reader and gather against B12X's
  Triton lookup bits, 40 fresh steps, rows that arrive while the main thread holds the
  GIL in a C call, a short read named on the next step, a timed-out wait that stops the
  GPU with its record (child process); the ported Engram model tests; the eager draft
  budget and the JIT monitor hook. The same r6k tests unchanged on dgx3 give the
  failures that predate this change.
- boots with the one-request probe: `fix-delay.json` (d1, d2: every read delayed
  200 ms, which hung 2 of 2 boots before) and `fix.json` (w1-w3).

Speed is judged by the release comparison of the candidate that carries this change.
