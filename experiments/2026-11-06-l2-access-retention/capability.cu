// SPDX-License-Identifier: Apache-2.0
// A bounded capability/cleanup probe, not a performance benchmark.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, \
               cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void read_region(const unsigned *in, unsigned *out) {
  unsigned i = threadIdx.x;
  out[i] = in[i] + 1;
}

void verify(unsigned *out) {
  unsigned values[32];
  CK(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < 32; ++i)
    if (values[i] != i + 1) std::exit(2);
}

int main() {
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0, runtime = 0;
  CK(cudaDriverGetVersion(&driver)); CK(cudaRuntimeGetVersion(&runtime));
  std::printf("{\"case\":\"device\",\"name\":\"%s\",\"major\":%d,\"minor\":%d,"
              "\"driver\":%d,\"runtime\":%d,\"l2_bytes\":%d,"
              "\"persisting_max_bytes\":%d,\"window_max_bytes\":%d}\n",
              prop.name, prop.major, prop.minor, driver, runtime, prop.l2CacheSize,
              prop.persistingL2CacheMaxSize, prop.accessPolicyMaxWindowSize);
  if (prop.major != 12 || prop.minor != 1) return 3;
  if (!prop.persistingL2CacheMaxSize || !prop.accessPolicyMaxWindowSize) return 4;
  size_t original = 0, actual = 0, free_before = 0, total = 0;
  CK(cudaDeviceGetLimit(&original, cudaLimitPersistingL2CacheSize));
  CK(cudaMemGetInfo(&free_before, &total));
  unsigned *in = nullptr, *out = nullptr;
  CK(cudaMalloc(&in, 128)); CK(cudaMalloc(&out, 128));
  unsigned values[32];
  for (unsigned i = 0; i < 32; ++i) values[i] = i;
  CK(cudaMemcpy(in, values, sizeof(values), cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1 << 20));
  CK(cudaDeviceGetLimit(&actual, cudaLimitPersistingL2CacheSize));
  std::printf("{\"case\":\"limit\",\"original_bytes\":%zu,"
              "\"requested_bytes\":1048576,\"applied_bytes\":%zu}\n", original, actual);
  if (!actual) return 5;
  cudaAccessPolicyWindow window{};
  window.base_ptr = in; window.num_bytes = 128; window.hitRatio = 1.0;
  window.hitProp = cudaAccessPropertyPersisting;
  window.missProp = cudaAccessPropertyNormal;
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = window;
  CK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
  cudaStreamAttrValue got{};
  CK(cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &got));
  if (got.accessPolicyWindow.num_bytes != 128 ||
      got.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting) return 6;
  read_region<<<1, 32, 0, stream>>>(in, out);
  CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream)); verify(out);

  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  read_region<<<1, 32, 0, stream>>>(in, out);
  CK(cudaStreamEndCapture(stream, &graph));
  size_t count = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  CK(cudaGraphGetNodes(graph, nodes.data(), &count));
  if (count != 1) return 7;
  cudaKernelNodeAttrValue node_attr{};
  CK(cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &node_attr));
  std::printf("{\"case\":\"capture\",\"window_bytes\":%zu,\"hit_property\":%d}\n",
              node_attr.accessPolicyWindow.num_bytes, int(node_attr.accessPolicyWindow.hitProp));
  const bool inherited = node_attr.accessPolicyWindow.num_bytes == 128 &&
      node_attr.accessPolicyWindow.hitProp == cudaAccessPropertyPersisting;
  // Prove the graph carries its own attribute after its stream is cleared.
  attr.accessPolicyWindow.num_bytes = 0;
  CK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, stream)); CK(cudaStreamSynchronize(stream)); verify(out);
  CK(cudaGraphExecDestroy(exec));
  node_attr.accessPolicyWindow = window;
  node_attr.accessPolicyWindow.hitRatio = 0.5;
  CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &node_attr));
  CK(cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &node_attr));
  if (node_attr.accessPolicyWindow.num_bytes != 128 ||
      node_attr.accessPolicyWindow.hitRatio != 0.5 ||
      node_attr.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting) return 8;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, stream)); CK(cudaStreamSynchronize(stream)); verify(out);
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));

  cudaLaunchAttribute launch_attr{};
  launch_attr.id = cudaLaunchAttributeAccessPolicyWindow;
  launch_attr.val.accessPolicyWindow = window;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1); config.blockDim = dim3(32);
  config.stream = stream; config.attrs = &launch_attr; config.numAttrs = 1;
  CK(cudaLaunchKernelEx(&config, read_region, in, out));
  CK(cudaStreamSynchronize(stream)); verify(out);
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CK(cudaLaunchKernelEx(&config, read_region, in, out));
  CK(cudaStreamEndCapture(stream, &graph));
  count = 1;
  CK(cudaGraphGetNodes(graph, nodes.data(), &count));
  CK(cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &node_attr));
  if (node_attr.accessPolicyWindow.num_bytes != 128) return 9;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, stream)); CK(cudaStreamSynchronize(stream)); verify(out);
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
  CK(cudaCtxResetPersistingL2Cache());
  CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, original));
  CK(cudaDeviceGetLimit(&actual, cudaLimitPersistingL2CacheSize));
  if (actual != original) return 10;
  CK(cudaStreamDestroy(stream)); CK(cudaFree(out)); CK(cudaFree(in));
  size_t free_after = 0;
  CK(cudaMemGetInfo(&free_after, &total));
  std::printf("{\"case\":\"verdict\",\"stream\":true,\"capture_inherits\":%s,"
              "\"graph_node\":true,\"extended_launch\":true,\"extended_capture\":true,"
              "\"exact\":true,\"restored_bytes\":%zu,\"free_before\":%zu,\"free_after\":%zu}\n",
              inherited ? "true" : "false", actual, free_before, free_after);
}
