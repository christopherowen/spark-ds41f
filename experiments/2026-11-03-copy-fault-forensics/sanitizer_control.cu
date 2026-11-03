// Diagnostic positive controls; never link into the inference runtime.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    std::exit(1);
  }
}

__global__ void read_value(const int* input, int* output, int index) {
  output[0] = input[index];
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "clean") != 0 &&
                    std::strcmp(argv[1], "oob") != 0 &&
                    std::strcmp(argv[1], "uninitialized") != 0)) {
    std::fprintf(stderr, "usage: %s clean|oob|uninitialized\n", argv[0]);
    return 2;
  }
  int* input = nullptr;
  int* output = nullptr;
  check(cudaMalloc(&input, sizeof(int)), "allocate input");
  check(cudaMalloc(&output, sizeof(int)), "allocate output");
  if (std::strcmp(argv[1], "uninitialized") != 0) {
    check(cudaMemset(input, 0, sizeof(int)), "initialize input");
  }
  // oob reads one whole element beyond this allocation. uninitialized reads
  // inside the allocation but intentionally omits its initialization.
  const int index = std::strcmp(argv[1], "oob") == 0 ? 1 : 0;
  read_value<<<1, 1>>>(input, output, index);
  check(cudaGetLastError(), "launch read_value");
  check(cudaDeviceSynchronize(), "synchronize read_value");
  int value = 0;
  check(cudaMemcpy(&value, output, sizeof(int), cudaMemcpyDeviceToHost), "read output");
  std::printf("mode=%s value=%d\n", argv[1], value);
  check(cudaFree(output), "free output");
  check(cudaFree(input), "free input");
  return 0;
}
