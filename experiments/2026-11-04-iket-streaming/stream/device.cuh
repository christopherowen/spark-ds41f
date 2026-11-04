// SPDX-License-Identifier: Apache-2.0
#include <cuda/atomic>
struct alignas(64) SparkTraceHeader {
  unsigned long long capacity, tail, head, status, waits, stop, reserved[2];
};
struct SparkTraceRecord {
  unsigned long long timestamp, grid, kernel, cta;
  unsigned warp, sm, event, payload, kind, reserved;
};
struct alignas(64) SparkTraceSlot {
  unsigned long long sequence;
  SparkTraceRecord record;
};
using SparkSysAtomic =
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
__device__ __forceinline__ unsigned long long spark_trace_name(const char *s) {
  unsigned long long n = 0;
  for (; *s; ++s)
    n = n * 10 + (*s - '0');
  return n;
}
__device__ __forceinline__ void spark_trace_emit(unsigned long long kernel,
                                                 unsigned kind, unsigned event,
                                                 unsigned payload) {
  unsigned linear =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const unsigned active = __activemask();
  if ((linear & 31) == 0) {
    auto h = reinterpret_cast<SparkTraceHeader *>(SPARK_TRACE_ADDRESS);
    if (SparkSysAtomic(h->status).load(cuda::memory_order_acquire))
      asm volatile("trap;");
    if (!SparkSysAtomic(h->stop).load(cuda::memory_order_acquire)) {
      unsigned long long ts, grid;
      unsigned sm;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
      asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid));
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      unsigned long long ticket =
          SparkSysAtomic(h->tail).fetch_add(1, cuda::memory_order_relaxed);
      auto slot =
          reinterpret_cast<SparkTraceSlot *>(h + 1) + ticket % h->capacity;
      bool waited = false;
      while (SparkSysAtomic(slot->sequence).load(cuda::memory_order_acquire) !=
             ticket) {
        if (!waited) {
          SparkSysAtomic(h->waits).fetch_add(1, cuda::memory_order_relaxed);
          waited = true;
        }
        if (SparkSysAtomic(h->status).load(cuda::memory_order_acquire))
          asm volatile("trap;");
        __nanosleep(256);
      }
      slot->record = SparkTraceRecord{
          ts,
          grid,
          kernel,
          (static_cast<unsigned long long>(blockIdx.z) * gridDim.y +
           blockIdx.y) *
                  gridDim.x +
              blockIdx.x,
          linear >> 5,
          sm,
          event,
          payload,
          kind,
          0};
      SparkSysAtomic(slot->sequence)
          .store(ticket + 1, cuda::memory_order_release);
    }
  }
  __syncwarp(active);
}
__device__ __forceinline__ void spark_trace_settle() {
  __shared__ unsigned word;
  unsigned value, addr = static_cast<unsigned>(__cvta_generic_to_shared(&word));
  asm volatile("ld.volatile.shared.u32 %0, [%1];"
               : "=r"(value)
               : "r"(addr)
               : "memory");
  asm volatile("" : : "r"(value) : "memory");
}
#define SPARK_TRACE_END(I) spark_trace_emit(__spark_trace_kernel, 2, 0, 0)
#define SPARK_TRACE_START(I, N)                                                \
  spark_trace_emit(__spark_trace_kernel, 1,                                    \
                   static_cast<unsigned>(spark_trace_name(N)), 0)
#define SPARK_TRACE_PICK(_1, _2, NAME, ...) NAME
#define TL_IKET_EVENT(...)                                                     \
  SPARK_TRACE_PICK(__VA_ARGS__, SPARK_TRACE_START, SPARK_TRACE_END)(__VA_ARGS__)
#define TL_IKET_EVENT_PAYLOAD_U32(I, P, N)                                     \
  spark_trace_emit(__spark_trace_kernel, 1,                                    \
                   static_cast<unsigned>(spark_trace_name(N)),                 \
                   static_cast<unsigned>(P))
#define TL_IKET_SETTLE() spark_trace_settle()
