// SPDX-License-Identifier: Apache-2.0
// Exact independent-stream LZ4 plus inverse bit permutation; no FP arithmetic.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TILE = 16384;
constexpr int PLANE = TILE / 8;
constexpr int WORDS = TILE / 8;

__device__ __forceinline__ bool decode_stream(
    const uint8_t *source, int length, uint8_t *destination, int capacity) {
    const unsigned lane = threadIdx.x & 31;
    int input = 0, output = 0;
    // Every lane follows the same bounded parser. Uniform header loads are
    // broadcast by the warp's memory transaction; copies use all its lanes.
    while (input < length) {
        const unsigned token = source[input++];
        int literals = token >> 4;
        if (literals == 15) {
            unsigned extra;
            do {
                if (input == length) return false;
                extra = source[input++];
                literals += extra;
                if (literals > capacity) return false;
            } while (extra == 255);
        }
        if (literals > length - input || literals > capacity - output) return false;
        for (int j = lane; j < literals; j += 32)
            destination[output + j] = source[input + j];
        __syncwarp();
        input += literals;
        output += literals;
        if (input == length) return output == capacity;
        if (length - input < 2) return false;
        const unsigned distance = source[input] | (unsigned(source[input + 1]) << 8);
        input += 2;
        if (distance == 0 || distance > unsigned(output)) return false;
        int match = (token & 15) + 4;
        if ((token & 15) == 15) {
            unsigned extra;
            do {
                if (input == length) return false;
                extra = source[input++];
                match += extra;
                if (match > capacity) return false;
            } while (extra == 255);
        }
        if (match > capacity - output) return false;
        const bool overlap = unsigned(match) > distance;
        const bool power_two = (distance & (distance - 1)) == 0;
        for (int j = lane; j < match; j += 32) {
            // An overlapping LZ4 match repeats its already-written distance
            // bytes. Read only that prefix, so warp copies have no write race.
            const unsigned position = !overlap ? unsigned(j)
                : power_two ? (unsigned(j) & (distance - 1)) : unsigned(j) % distance;
            destination[output + j] = destination[output - distance + position];
        }
        __syncwarp();
        output += match;
    }
    return false;  // A conforming independent block ends after its literals.
}

__device__ __forceinline__ uint64_t transpose8(uint64_t word) {
    uint64_t change = (word ^ (word >> 7)) & 0x00AA00AA00AA00AAULL;
    word ^= change ^ (change << 7);
    change = (word ^ (word >> 14)) & 0x0000CCCC0000CCCCULL;
    word ^= change ^ (change << 14);
    change = (word ^ (word >> 28)) & 0x00000000F0F0F0F0ULL;
    return word ^ change ^ (change << 28);
}

__device__ __forceinline__ uint64_t original_word(const uint8_t *planes, int group) {
    uint64_t word = 0;
    #pragma unroll
    for (int plane = 0; plane < 8; ++plane)
        word |= uint64_t(planes[plane * PLANE + group]) << (plane * 8);
    return transpose8(word);
}

template <bool Consume>
__global__ void decode_tiles(const uint8_t *payload, const int64_t *starts,
    const uint8_t *modes, uint64_t *output, int32_t *checksums) {
    __shared__ uint64_t tile[WORDS];
    __shared__ unsigned partial[16];
    const int t = blockIdx.x;
    const uint8_t *data = payload + starts[t];
    const bool encoded = modes[t] != 0;
    auto *planes = reinterpret_cast<uint8_t *>(tile);
    if (encoded) {
        const int chunk = reinterpret_cast<const uint16_t *>(data)[3];
        const int streams = TILE / chunk;
        const int lane = threadIdx.x & 31;
        for (int stream = threadIdx.x / 32; stream < streams; stream += blockDim.x / 32) {
            const auto *entry = reinterpret_cast<const uint16_t *>(data + 16) + stream * 2;
            const uint8_t *source = data + entry[0];
            uint8_t *destination = planes + stream * chunk;
            const int length = entry[1] & 0x7fff;
            if (entry[1] & 0x8000) {
                for (int word = lane; word < chunk / 8; word += 32)
                    reinterpret_cast<uint64_t *>(destination)[word]
                        = reinterpret_cast<const uint64_t *>(source)[word];
            } else if (!decode_stream(source, length, destination, chunk)) {
                // Host validation precedes upload. A violated stream contract
                // fails the job instead of consuming partial shared bytes.
                asm volatile("trap;");
                return;
            }
        }
    } else {
        for (int index = threadIdx.x; index < WORDS; index += blockDim.x)
            tile[index] = reinterpret_cast<const uint64_t *>(data)[index];
    }
    __syncthreads();
    if constexpr (Consume) {
        unsigned value = 0;
        // The representation and eight-byte shared consumer are an explicit
        // interaction. Both arms consume the same original bytes; coded data
        // stays in shared bitplanes until its register-sized consumption.
        for (int index = threadIdx.x; index < WORDS; index += blockDim.x) {
            const int position = (index + 17) & (WORDS - 1);
            const uint64_t word = encoded ? original_word(planes, position) : tile[position];
            value ^= unsigned(word) ^ unsigned(word >> 32);
        }
        value = __reduce_xor_sync(0xffffffff, value);
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = value;
        __syncthreads();
        if (threadIdx.x == 0) {
            value = 0;
            for (int warp = 0; warp < blockDim.x / 32; ++warp) value ^= partial[warp];
            checksums[t] = static_cast<int32_t>(value);
        }
    } else {
        for (int index = threadIdx.x; index < WORDS; index += blockDim.x)
            output[t * WORDS + index] = encoded ? original_word(planes, index) : tile[index];
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
    // Every bounded stream and aligned extent is CPU-validated before upload.
    auto *out = reinterpret_cast<uint64_t *>(output.data_ptr<uint8_t>());
    if (consume)
        decode_tiles<true><<<count, threads, 0, stream>>>(payload.data_ptr<uint8_t>(),
            starts.data_ptr<int64_t>(), modes.data_ptr<uint8_t>(), out, checksums.data_ptr<int32_t>());
    else
        decode_tiles<false><<<count, threads, 0, stream>>>(payload.data_ptr<uint8_t>(),
            starts.data_ptr<int64_t>(), modes.data_ptr<uint8_t>(), out, checksums.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
