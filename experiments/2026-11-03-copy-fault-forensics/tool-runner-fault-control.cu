// Diagnostic control only. Never link into or alter the inference runtime.
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

__device__ volatile float control_input[2];
__device__ volatile float control_output;
__device__ volatile unsigned int control_bits;
__device__ volatile int control_stage;

__global__ void control_producer() {
  const float value = control_input[0] / control_input[1];
  control_output = value;
  control_bits = __float_as_uint(value);
  control_stage = 1;
}

__global__ void control_consumer() {
  control_stage = 2;
  assert(isfinite(control_output) && "deliberate nonfinite control");
}

static bool check(cudaError_t code, const char* operation) {
  if (code == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s (%d)\n", operation,
               cudaGetErrorString(code), static_cast<int>(code));
  return false;
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "clean") != 0 &&
                    std::strcmp(argv[1], "nan") != 0)) {
    std::fprintf(stderr, "usage: %s clean|nan\n", argv[0]);
    return 2;
  }
  const bool fault = std::strcmp(argv[1], "nan") == 0;
  const float input[2] = {fault ? 0.0f : 1.0f, fault ? 0.0f : 1.0f};
  if (!check(cudaMemcpyToSymbol(control_input, input, sizeof(input)), "initialize input")) return 3;
  cudaStream_t producer, consumer;
  cudaEvent_t ready;
  if (!check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), "producer stream") ||
      !check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "consumer stream") ||
      !check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "ready event")) return 3;
  nvtxRangePushA(fault ? "control/nan/original-values" : "control/clean/original-values");
  nvtxRangePushA("control/producer");
  control_producer<<<1, 1, 0, producer>>>();
  if (!check(cudaGetLastError(), "launch producer") ||
      !check(cudaEventRecord(ready, producer), "publish producer")) return 3;
  nvtxRangePop();
  nvtxRangePushA("control/consumer-after-event");
  if (!check(cudaStreamWaitEvent(consumer, ready), "consumer waits")) return 3;
  control_consumer<<<1, 1, 0, consumer>>>();
  if (!check(cudaGetLastError(), "launch consumer")) return 3;
  nvtxRangePop();
  const cudaError_t result = cudaDeviceSynchronize();
  nvtxRangePop();
  if (!check(result, "synchronize control")) return 3;
  float output;
  unsigned int bits;
  int stage;
  if (!check(cudaMemcpyFromSymbol(&output, control_output, sizeof(output)), "read output") ||
      !check(cudaMemcpyFromSymbol(&bits, control_bits, sizeof(bits)), "read bits") ||
      !check(cudaMemcpyFromSymbol(&stage, control_stage, sizeof(stage)), "read stage")) return 3;
  std::printf("mode=%s input=%g/%g output=%g bits=0x%08x stage=%d\n",
              argv[1], input[0], input[1], output, bits, stage);
  cudaEventDestroy(ready);
  cudaStreamDestroy(consumer);
  cudaStreamDestroy(producer);
  return output == 1.0f && bits == 0x3f800000u && stage == 2 ? 0 : 4;
}
