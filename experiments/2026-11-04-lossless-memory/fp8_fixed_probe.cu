// SPDX-License-Identifier: Apache-2.0
// Experimental exact-byte consumer; no floating-point arithmetic or GEMM.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TILE = 16384;
constexpr int WORDS = TILE / 4;
constexpr int LOW = 16;
constexpr int TAGS = LOW + TILE / 2;
constexpr int OFFSETS = TAGS + TILE / 4;
constexpr int EXCEPTIONS = OFFSETS + TILE / 128 * 2;

__device__ __forceinline__ uint32_t decode_word(const uint8_t *data, int word) {
    const unsigned lane = threadIdx.x & 31;
    // Eight lanes share a 64-bit word of 32 two-bit tags. A complete warp
    // covers one 128-value exception chunk. Prefix counts never cross chunks.
    const uint64_t tags = reinterpret_cast<const uint64_t *>(data + TAGS)[word / 8];
    const uint64_t mask = tags & (tags >> 1) & 0x5555555555555555ULL;
    const unsigned count = __popcll(mask);
    unsigned prefix = count;
    unsigned preceding = __shfl_up_sync(0xffffffff, prefix, 8);
    if (lane >= 8) prefix += preceding;
    preceding = __shfl_up_sync(0xffffffff, prefix, 16);
    if (lane >= 16) prefix += preceding;
    const unsigned shift = (word & 7) * 8;
    const uint64_t below = shift == 0 ? 0 : ((1ULL << shift) - 1);
    unsigned position = reinterpret_cast<const uint16_t *>(data + OFFSETS)[word / 32]
                      + prefix - count + __popcll(mask & below);
    const unsigned low = reinterpret_cast<const uint16_t *>(data + LOW)[word];
    const unsigned book = reinterpret_cast<const uint16_t *>(data + 4)[0];
    uint32_t value = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned tag = (tags >> (shift + j * 2)) & 3;
        unsigned exponent;
        if (tag == 3) {
            exponent = (data[EXCEPTIONS + position / 2] >> (4 * (position & 1))) & 15;
            ++position;
        } else {
            exponent = (book >> (4 * tag)) & 15;
        }
        const unsigned a = (low >> (4 * j)) & 15;
        value |= ((a & 7) | ((a & 8) << 4) | (exponent << 3)) << (8 * j);
    }
    return value;
}

template <bool Consume>
__global__ void decode_tiles(const uint8_t *payload, const int64_t *starts,
    const uint8_t *modes, uint32_t *output, int32_t *checksums) {
    __shared__ uint32_t tile[WORDS];
    __shared__ unsigned partial[16];
    const int t = blockIdx.x;
    const uint8_t *data = payload + starts[t];
    for (int index = threadIdx.x; index < WORDS; index += blockDim.x) {
        const uint32_t value = modes[t] == 0
            ? reinterpret_cast<const uint32_t *>(data)[index] : decode_word(data, index);
        if constexpr (Consume) tile[index] = value;
        else output[t * WORDS + index] = value;
    }
    if constexpr (Consume) {
        __syncthreads();
        unsigned value = 0;
        // Both arms use the same word-sized global/shared traffic and consumer.
        // The representation and consumer granularity are an explicit interaction.
        for (int index = threadIdx.x; index < WORDS; index += blockDim.x)
            value ^= tile[(index + 17) & (WORDS - 1)];
        value = __reduce_xor_sync(0xffffffff, value);
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = value;
        __syncthreads();
        if (threadIdx.x == 0) {
            value = 0;
            for (int warp = 0; warp < blockDim.x / 32; ++warp) value ^= partial[warp];
            checksums[t] = static_cast<int32_t>(value);
        }
    }
}

void run(torch::Tensor payload, torch::Tensor starts, torch::Tensor modes,
    torch::Tensor output, torch::Tensor checksums, int64_t threads, bool consume) {
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid thread count");
    TORCH_CHECK(modes.numel() > 0 && modes.numel() <= 65536, "invalid tile count");
    TORCH_CHECK(payload.is_cuda(), "CUDA tensors required");
    const auto device = payload.device();
    for (const auto &tensor : {payload, starts, modes, output, checksums})
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
                    "probe tensors must be contiguous on one CUDA device");
    const auto count = modes.numel();
    TORCH_CHECK(starts.numel() == count && output.numel() == count * TILE
                && checksums.numel() == count, "probe shape mismatch");
    c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    // The host validates every frame and its aligned extent before upload.
    auto *out = reinterpret_cast<uint32_t *>(output.data_ptr<uint8_t>());
    if (consume)
        decode_tiles<true><<<count, threads, 0, stream>>>(payload.data_ptr<uint8_t>(),
            starts.data_ptr<int64_t>(), modes.data_ptr<uint8_t>(), out, checksums.data_ptr<int32_t>());
    else
        decode_tiles<false><<<count, threads, 0, stream>>>(payload.data_ptr<uint8_t>(),
            starts.data_ptr<int64_t>(), modes.data_ptr<uint8_t>(), out, checksums.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
