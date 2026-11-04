// SPDX-License-Identifier: Apache-2.0
// Experimental exact-byte decoder; this is not a serving GEMM or storage ABI.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TILE = 16384;

template <bool Consume>
__global__ void decode_tiles(const uint8_t *payload, const int64_t *starts,
    const uint8_t *low, const uint8_t *tree, const uint16_t *quick,
    const int32_t *streams, const uint8_t *modes, uint8_t *output,
    int32_t *checksums, int lanes) {
    __shared__ uint8_t tile[TILE];
    __shared__ unsigned partial[4];
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const uint8_t *data = payload + starts[t];
    const uint8_t *stream = data + streams[t * lanes + lane];
    unsigned position = 0;
    for (int index = lane; index < TILE; index += lanes) {
        uint8_t value;
        if (modes[t] == 0) {
            value = data[index];
        } else {
            const unsigned byte = position >> 3;
            const unsigned word = (unsigned(stream[byte]) << 16) |
                                  (unsigned(stream[byte + 1]) << 8) | stream[byte + 2];
            const unsigned prefix = (word >> (16 - (position & 7))) & 255;
            const unsigned entry = quick[t * 256 + prefix];
            unsigned length = entry >> 4;
            unsigned exponent = entry & 15;
            if (length == 0) {
                unsigned node = 0;
                // Inputs have already passed the CPU bounds/integrity check.
                // Rare codes longer than eight bits follow the exact tree.
                for (length = 1; length <= 15; ++length) {
                    const unsigned p = position + length - 1;
                    const unsigned bit = (stream[p >> 3] >> (7 - (p & 7))) & 1;
                    node = tree[t * 32 + node * 2 + bit];
                    if (node == 255) { asm("trap;"); return; }
                    if (node & 128) {
                        exponent = node & 15;
                        break;
                    }
                }
                if (length > 15) { asm("trap;"); return; }
            }
            position += length;
            const uint8_t packed = low[t * (TILE / 2) + index / 2];
            const unsigned a = (packed >> (4 * (index & 1))) & 15;
            value = (a & 7) | ((a & 8) << 4) | (exponent << 3);
        }
        if constexpr (Consume) tile[index] = value;
        else output[t * TILE + index] = value;
    }
    if constexpr (Consume) {
        __syncthreads();
        unsigned value = 0;
        // Consume a different lane's shared bytes so reconstruction is live.
        for (int index = lane; index < TILE; index += lanes)
            value ^= unsigned(tile[(index + 17) & (TILE - 1)]) << (8 * (index & 3));
        value = __reduce_xor_sync(0xffffffff, value);
        if ((lane & 31) == 0) partial[lane >> 5] = value;
        __syncthreads();
        if (lane == 0) {
            value = 0;
            for (int warp = 0; warp < lanes / 32; ++warp) value ^= partial[warp];
            checksums[t] = static_cast<int32_t>(value);
        }
    }
}

void run(torch::Tensor payload, torch::Tensor starts, torch::Tensor low,
    torch::Tensor tree, torch::Tensor quick, torch::Tensor streams,
    torch::Tensor modes, torch::Tensor output, torch::Tensor checksums,
    int64_t lanes, bool consume) {
    TORCH_CHECK(lanes == 32 || lanes == 64 || lanes == 128, "invalid lane count");
    TORCH_CHECK(modes.numel() > 0 && modes.numel() <= 65536, "invalid tile count");
    const auto device = payload.device();
    TORCH_CHECK(payload.is_cuda(), "CUDA tensors are required");
    for (const auto &tensor : {payload, starts, low, tree, quick, streams, modes, output, checksums})
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
                    "all probe tensors must be contiguous on one CUDA device");
    const auto count = modes.numel();
    TORCH_CHECK(starts.numel() == count && low.numel() == count * TILE / 2 &&
                tree.numel() == count * 32 && quick.numel() == count * 256 &&
                streams.numel() == count * lanes && output.numel() == count * TILE &&
                checksums.numel() == count, "probe shape mismatch");
    c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    if (consume)
        decode_tiles<true><<<count, lanes, 0, stream>>>(payload.data_ptr<uint8_t>(), starts.data_ptr<int64_t>(),
            low.data_ptr<uint8_t>(), tree.data_ptr<uint8_t>(), quick.data_ptr<uint16_t>(),
            streams.data_ptr<int32_t>(), modes.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            checksums.data_ptr<int32_t>(), static_cast<int>(lanes));
    else
        decode_tiles<false><<<count, lanes, 0, stream>>>(payload.data_ptr<uint8_t>(), starts.data_ptr<int64_t>(),
            low.data_ptr<uint8_t>(), tree.data_ptr<uint8_t>(), quick.data_ptr<uint16_t>(),
            streams.data_ptr<int32_t>(), modes.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            checksums.data_ptr<int32_t>(), static_cast<int>(lanes));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
