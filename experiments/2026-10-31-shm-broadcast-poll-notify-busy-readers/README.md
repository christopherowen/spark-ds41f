# Avoid Unnecessary TCP Socket Notify When `shm_broadcast` Readers Are Busy-Polling (`#61844`)

## Upstream Source

- Upstream commit: `vllm-project/vllm@83c078d955` (`[Perf] Avoid unnecessary TCP socket notify when shm_broadcast readers are already busy-polling (#61844)`)

## Problem

When `VLLM_SHM_BROADCAST_POLL_NOTIFY_ENABLED=1` is used, `ShmRingBufferIO.commit_write()` previously notified readers over the ZMQ/TCP wakeup socket unless all readers had already flipped their state back to `0` at the exact instant `commit_write()` ran. During active decode bursts, readers are already awake and busy-polling in their spin window (`self._last_write_time` within `VLLM_RINGBUFFER_BUSY_WAIT_MS`), so sending a TCP socket notification on every broadcast step adds redundant kernel/syscall overhead on the main scheduler/worker IPC path.

## Changes

- **`vllm/distributed/device_communicators/shm_broadcast.py`**:
  - Track `self._last_write_time` in `ShmRingBufferIO.commit_write()`.
  - Skip the `poll_notify_socket.send()` call when the time elapsed since the previous write is less than `_SpinTimer.SPIN_TIME_S` (`VLLM_RINGBUFFER_BUSY_WAIT_MS / 1000.0`), since readers are guaranteed to still be in their active spin window.
- **`tests/distributed/test_shm_broadcast.py`**:
  - Add `test_poll_notify_skips_send_when_readers_are_busy_waiting` verifying that back-to-back writes within the spin window suppress redundant socket notifications while idle gaps still trigger wakeup notifications.

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/distributed/test_shm_broadcast.py -k "not test_broadcast"` (`31 passed`).
