// Numerical instrumentation coverage control, not a model reproducer.
// A full warp avoids undefined inactive-lane reads in third-party checkers.
// Each lane reads and writes its own initialized float, with no shared writes.
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

__global__ void nvbit_warp_numerical_control(const float* input, float* output) {
  const unsigned int lane = threadIdx.x;
  float result;
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(input[lane]));
  output[lane] = result;
}

int main(int argc, char** argv) {
  if (argc != 3 || (std::strcmp(argv[1], "clean") &&
                    std::strcmp(argv[1], "nan")) ||
      (std::strcmp(argv[2], "direct") && std::strcmp(argv[2], "extended") &&
       std::strcmp(argv[2], "capture") && std::strcmp(argv[2], "graph"))) {
    std::fprintf(stderr, "usage: %s clean|nan direct|extended|capture|graph\n", argv[0]);
    return 2;
  }
  const bool expect_nan = std::strcmp(argv[1], "nan") == 0;
  float input[32], output[32] = {};
  for (float& value : input) value = expect_nan ? -4.0f : 4.0f;
  float *device_input = nullptr, *device_output = nullptr;
  check(cudaMalloc(&device_input, sizeof(input)), "allocate input");
  check(cudaMalloc(&device_output, sizeof(output)), "allocate output");
  check(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice),
        "initialize input");
  check(cudaMemcpy(device_output, output, sizeof(output), cudaMemcpyHostToDevice),
        "initialize output");
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (!std::strcmp(argv[2], "extended")) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(32);
    config.stream = stream;
    check(cudaLaunchKernelEx(&config, nvbit_warp_numerical_control,
                            device_input, device_output), "extended launch");
  } else if (!std::strcmp(argv[2], "graph")) {
    check(cudaGraphCreate(&graph, 0), "create graph");
    void* arguments[] = {&device_input, &device_output};
    cudaKernelNodeParams parameters{};
    parameters.func = reinterpret_cast<void*>(nvbit_warp_numerical_control);
    parameters.gridDim = dim3(1);
    parameters.blockDim = dim3(32);
    parameters.kernelParams = arguments;
    cudaGraphNode_t node;
    check(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &parameters), "add graph kernel");
  } else {
    if (!std::strcmp(argv[2], "capture"))
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    nvbit_warp_numerical_control<<<1, 32, 0, stream>>>(device_input, device_output);
    check(cudaGetLastError(), "launch control");
    if (!std::strcmp(argv[2], "capture"))
      check(cudaStreamEndCapture(stream, &graph), "end capture");
  }
  if (graph) {
    check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
    check(cudaGraphLaunch(executable, stream), "launch graph");
  }
  check(cudaStreamSynchronize(stream), "finish control");
  check(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost),
        "read output");
  if (executable) check(cudaGraphExecDestroy(executable), "destroy executable");
  if (graph) check(cudaGraphDestroy(graph), "destroy graph");
  check(cudaStreamDestroy(stream), "destroy stream");
  check(cudaFree(device_output), "free output");
  check(cudaFree(device_input), "free input");
  unsigned int correct = 0;
  for (float value : output)
    correct += expect_nan ? std::isnan(value) : value == 2.0f;
  std::printf("mode=%s launch=%s correct_lanes=%u/32 first_output=%g\n",
              argv[1], argv[2], correct, output[0]);
  return correct == 32 ? 0 : 1;
}
