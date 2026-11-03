// Standalone diagnostic controls; never link into the inference runtime.
// Build: nvcc -lineinfo -O0 -arch=sm_121 kernel_sync_controls.cu -o kernel_sync_controls
// Run each mode in a separate process, under an external 30-second timeout:
//   compute-sanitizer --tool racecheck --error-exitcode 86 ./kernel_sync_controls race-clean
//   compute-sanitizer --tool racecheck --error-exitcode 86 ./kernel_sync_controls race
//   compute-sanitizer --tool synccheck --error-exitcode 86 ./kernel_sync_controls sync-clean
//   compute-sanitizer --tool synccheck --error-exitcode 86 ./kernel_sync_controls sync-invalid
// Expected: clean modes exit 0 with no findings; positive modes report the
// indicated hazard and exit 86. A different outcome is a retained control
// failure, not evidence that the application is free from that error class.
// No missing participant is named by the warp control's mask. The deliberate
// error is an extra caller outside the mask; avoid a missing-thread barrier
// that can deadlock. This is an invalid-mask control, not block-divergence proof.
// NVIDIA documents this specific synccheck error at:
// https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#understanding-synccheck-reports

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

template <bool Race>
__global__ void shared_write_control(int* output) {
  __shared__ volatile int shared_value;
  if (threadIdx.x == 0) shared_value = 0;
  __syncthreads();
  if (threadIdx.x == 0) shared_value = 1;
  // Without this barrier, different warps write the same shared address
  // without a happens-before edge: an intentional WAW hazard.
  if (!Race) __syncthreads();
  if (threadIdx.x == 32) shared_value = 33;
  __syncthreads();
  if (threadIdx.x == 0) output[0] = shared_value;
}

template <bool Invalid>
__global__ void warp_mask_control(int* output) {
  const unsigned lane = threadIdx.x;
  if (lane < 16 || (Invalid && lane == 16)) {
    // The 16 named lanes all arrive. Only the positive control adds lane16,
    // whose own bit is absent: synccheck should report Invalid arguments.
    __syncwarp(0x0000ffffu);
  }
  output[lane] = static_cast<int>(lane);
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "race-clean") != 0 &&
                    std::strcmp(argv[1], "race") != 0 &&
                    std::strcmp(argv[1], "sync-clean") != 0 &&
                    std::strcmp(argv[1], "sync-invalid") != 0)) {
    std::fprintf(stderr, "usage: %s race-clean|race|sync-clean|sync-invalid\n", argv[0]);
    return 2;
  }
  int* output = nullptr;
  check(cudaMalloc(&output, 32 * sizeof(int)), "allocate output");
  check(cudaMemset(output, 0, 32 * sizeof(int)), "initialize output");
  const bool race_clean = std::strcmp(argv[1], "race-clean") == 0;
  const bool race = std::strcmp(argv[1], "race") == 0;
  const bool sync_clean = std::strcmp(argv[1], "sync-clean") == 0;
  if (race_clean) shared_write_control<false><<<1, 64>>>(output);
  else if (race) shared_write_control<true><<<1, 64>>>(output);
  else if (sync_clean) warp_mask_control<false><<<1, 32>>>(output);
  else warp_mask_control<true><<<1, 32>>>(output);
  check(cudaGetLastError(), "launch control");
  check(cudaDeviceSynchronize(), "synchronize control");
  int values[32] = {};
  check(cudaMemcpy(values, output, sizeof(values), cudaMemcpyDeviceToHost), "read output");
  bool valid_clean_result = !race_clean || values[0] == 33;
  if (sync_clean) {
    for (int lane = 0; lane < 32; ++lane) valid_clean_result &= values[lane] == lane;
  }
  std::printf("mode=%s first_value=%d clean_result_valid=%s\n", argv[1], values[0],
              valid_clean_result ? "true" : "false");
  check(cudaFree(output), "free output");
  return valid_clean_result ? 0 : 1;
}
