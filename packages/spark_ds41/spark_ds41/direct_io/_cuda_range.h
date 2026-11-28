/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the b12x project
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Whether a pointer range lies in CUDA device allocations of one device.
 *
 * Ported from local-inference-lab/b12x b12x/loader/_cuda_range.h at 236ddff04a5f
 * (Apache-2.0). Changed: the include guard's name, and external dma-buf device
 * mappings under CUDA 13.5 forward compatibility.
 */
#ifndef VLLM_DIRECT_IO_CUDA_RANGE_H
#define VLLM_DIRECT_IO_CUDA_RANGE_H
#include <cuda.h>

/* External dma-buf device mappings (cuImportExternalMemory +
 * cuExternalMemoryGetMappedBuffer, such as display_carveout's DRM dumb buffers)
 * report CU_MEMORYTYPE_HOST with no host pointer and base == 0 in
 * cuMemGetAddressRange under CUDA 13.5 forward compatibility (cuda-compat-13-5
 * 620.x over a 580.x host driver), while CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
 * CU_POINTER_ATTRIBUTE_RANGE_START_ADDR and CU_POINTER_ATTRIBUTE_RANGE_SIZE
 * still identify the mapped device buffer. */
static bool is_device_only_mapping(uintptr_t pointer, CUmemorytype kind) {
    if (kind == CU_MEMORYTYPE_DEVICE) return true;
    if (kind != CU_MEMORYTYPE_HOST) return false;
    void *host_ptr = NULL;
    CUdeviceptr dev_ptr = 0;
    return cuPointerGetAttribute(&host_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, pointer) != CUDA_SUCCESS &&
           cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, pointer) == CUDA_SUCCESS &&
           dev_ptr == (CUdeviceptr)pointer;
}

/* Expandable tensors can span several adjacent CUDA mappings. */
static bool device_range(uintptr_t pointer, uint64_t bytes, int device) {
    if (bytes > UINTPTR_MAX - pointer) return false;
    do {
        CUmemorytype kind;
        int ordinal;
        CUdeviceptr base;
        size_t size;
        if (cuPointerGetAttribute(&kind, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer) != CUDA_SUCCESS ||
            !is_device_only_mapping(pointer, kind) ||
            cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer) != CUDA_SUCCESS ||
            ordinal != device ||
            cuMemGetAddressRange(&base, &size, pointer) != CUDA_SUCCESS) return false;
        if (base == 0 && kind == CU_MEMORYTYPE_HOST) {
            CUdeviceptr range_start = 0;
            size_t range_size = 0;
            if (cuPointerGetAttribute(&range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, pointer) != CUDA_SUCCESS ||
                cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, pointer) != CUDA_SUCCESS ||
                range_start == 0 ||
                range_size != size) return false;
            base = range_start;
        }
        if ((CUdeviceptr)pointer < base || (CUdeviceptr)pointer - base >= size) return false;
        size_t available = size - (size_t)((CUdeviceptr)pointer - base);
        if (bytes <= available) return true;
        pointer += available;
        bytes -= available;
    } while (bytes);
    return true;
}
#endif
