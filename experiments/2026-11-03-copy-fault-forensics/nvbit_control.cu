// Standalone instrumentation control, not a CopyDrafter reproducer.
// All device inputs and outputs are initialized; one thread touches two floats.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    std::exit(2);
  }
}

__global__ void nvbit_numerical_control(const float* input, float* output) {
  float result;
  // Runtime-loaded finite input prevents constant folding. Negative input
  // produces NaN at a floating-point instruction, not an invalid memory read.
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(*input));
  *output = result;
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "clean") &&
                    std::strcmp(argv[1], "nan"))) {
    std::fprintf(stderr, "usage: %s clean|nan\n", argv[0]);
    return 2;
  }
  const bool expect_nan = std::strcmp(argv[1], "nan") == 0;
  const float input = expect_nan ? -4.0f : 4.0f;
  float output = 0.0f;
  float *device_input = nullptr, *device_output = nullptr;
  check(cudaMalloc(&device_input, sizeof(float)), "allocate input");
  check(cudaMalloc(&device_output, sizeof(float)), "allocate output");
  check(cudaMemcpy(device_input, &input, sizeof(float), cudaMemcpyHostToDevice),
        "initialize input");
  check(cudaMemset(device_output, 0, sizeof(float)), "initialize output");
  nvbit_numerical_control<<<1, 1>>>(device_input, device_output);
  check(cudaGetLastError(), "launch control");
  check(cudaDeviceSynchronize(), "finish control");
  check(cudaMemcpy(&output, device_output, sizeof(float), cudaMemcpyDeviceToHost),
        "read output");
  check(cudaFree(device_output), "free output");
  check(cudaFree(device_input), "free input");
  const bool passed = expect_nan ? std::isnan(output) : output == 2.0f;
  std::printf("mode=%s input=%g output=%g expected_result=%s\n", argv[1], input,
              output, passed ? "yes" : "no");
  return passed ? 0 : 1;
}
