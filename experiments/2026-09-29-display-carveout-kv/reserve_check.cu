// Correctness of the dma-buf-imported display carve-out: GPU writes, CPU and GPU read back.
#include <cuda.h>
#include <cuda_runtime.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
__global__ void fill(uint64_t *p, size_t n, uint64_t salt) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += gridDim.x * (size_t)blockDim.x)
        p[i] = (i * 0x9E3779B97F4A7C15ull) ^ salt;
}
__global__ void verify(const uint64_t *p, size_t n, uint64_t salt, unsigned long long *bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += gridDim.x * (size_t)blockDim.x)
        if (p[i] != ((i * 0x9E3779B97F4A7C15ull) ^ salt)) atomicAdd(bad, 1ull);
}
int main() {
    size_t bytes = 2032ul << 20, n = bytes / 8;
    cudaFree(0);
    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    struct drm_mode_create_dumb c = {0}; c.width = 4096; c.bpp = 32; c.height = bytes / (4096 * 4);
    if (ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c)) { perror("create"); return 1; }
    struct drm_mode_map_dumb m = {0}; m.handle = c.handle; ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m);
    volatile uint64_t *cpu = (volatile uint64_t *)mmap(0, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    struct drm_prime_handle ph = {0}; ph.handle = c.handle; ph.flags = DRM_CLOEXEC | DRM_RDWR;
    ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph);
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd = {}; hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
    hd.handle.fd = ph.fd; hd.size = c.size;
    CUexternalMemory em; if (cuImportExternalMemory(&em, &hd)) { printf("import failed\n"); return 1; }
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = {}; bd.size = c.size; CUdeviceptr d;
    if (cuExternalMemoryGetMappedBuffer(&d, em, &bd)) { printf("map failed\n"); return 1; }
    unsigned long long *bad; cudaMallocManaged(&bad, 8);
    for (uint64_t salt : {0x1234ull, 0xdeadbeefcafef00dull}) {
        *bad = 0;
        fill<<<1024, 512>>>((uint64_t *)d, n, salt); cudaDeviceSynchronize();
        size_t cpu_bad = 0, probes[] = {0, 1, 4095, n / 3, n / 2 + 7, n - 1};
        for (size_t i : probes) if (cpu[i] != ((i * 0x9E3779B97F4A7C15ull) ^ salt)) cpu_bad++;
        verify<<<1024, 512>>>((const uint64_t *)d, n, salt, bad); cudaDeviceSynchronize();
        printf("salt %#llx: gpu mismatches %llu of %zu, cpu probe mismatches %zu of 6\n",
               (unsigned long long)salt, *bad, n, cpu_bad);
    }
    // CPU write, GPU read (coherence the other way).
    cpu[12345] = 0x5555aaaa5555aaaaull; __sync_synchronize();
    uint64_t back = 0; cudaMemcpy(&back, (void *)(d + 12345 * 8), 8, cudaMemcpyDeviceToHost);
    printf("cpu->gpu word: %s\n", back == 0x5555aaaa5555aaaaull ? "ok" : "MISMATCH");
    printf("cudaMemset: %s\n", cudaGetErrorString(cudaMemset((void *)d, 0, c.size)));
    cudaDeviceSynchronize(); printf("after memset cpu[n-1]=%llu\n", (unsigned long long)cpu[n - 1]);
    cuDestroyExternalMemory(em); close(ph.fd);
    struct drm_mode_destroy_dumb dd = {c.handle}; ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dd); close(fd);
    return 0;
}
