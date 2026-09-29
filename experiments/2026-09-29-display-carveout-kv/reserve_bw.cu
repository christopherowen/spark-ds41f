// GPU bandwidth of the GB10 display carve-out (DRM dumb buffer registered as
// device-mapped I/O memory) against ordinary cudaMalloc memory.
// Build: nvcc -O3 -arch=native reserve_bw.cu -lcuda -o reserve_bw
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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void rd(const uint4 *p, size_t n, uint4 *sink) {
    uint4 a = {0, 0, 0, 0};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += gridDim.x * (size_t)blockDim.x) {
        uint4 v = p[i]; a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if (a.x == 0xdeadbeef) *sink = a;
}

__global__ void wr(uint4 *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += gridDim.x * (size_t)blockDim.x)
        p[i] = make_uint4(i, i, i, i);
}

// Block-strided gather: each warp reads one contiguous span of `span` bytes at
// a pseudo-random block, like paged KV reads.
__global__ void gather(const uint4 *p, size_t nblocks, size_t span16, uint4 *sink, int iters) {
    uint4 a = {0, 0, 0, 0};
    size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    size_t nwarps = gridDim.x * (size_t)blockDim.x / 32;
    for (int it = 0; it < iters; it++) {
        size_t b = ((warp + it * nwarps) * 2654435761ull) % nblocks;
        const uint4 *q = p + b * span16;
        for (size_t i = lane; i < span16; i += 32) { uint4 v = q[i]; a.x ^= v.x; a.y ^= v.y; }
    }
    if (a.x == 0xdeadbeef) *sink = a;
}

static float timed(void (*launch)(void *), void *arg, int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    launch(arg); CK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < reps; i++) launch(arg);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); return ms / reps;
}

struct Arg { void *p; size_t bytes; uint4 *sink; };
static void launch_rd(void *a) { Arg *x = (Arg *)a; rd<<<1024, 512>>>((uint4 *)x->p, x->bytes / 16, x->sink); }
static void launch_wr(void *a) { Arg *x = (Arg *)a; wr<<<1024, 512>>>((uint4 *)x->p, x->bytes / 16); }
static void launch_gather(void *a) {
    Arg *x = (Arg *)a; size_t span = 64 * 1024;  // 64 KiB spans
    gather<<<1024, 256>>>((uint4 *)x->p, x->bytes / span, span / 16, x->sink, 4);
}

static void report(const char *name, void *p, size_t bytes, uint4 *sink) {
    Arg a = {p, bytes, sink};
    float r = timed(launch_rd, &a, 10), w = timed(launch_wr, &a, 10), g = timed(launch_gather, &a, 10);
    size_t gbytes = (size_t)1024 * 256 / 32 * 4 * 64 * 1024;
    printf("%-10s read %6.1f GB/s  write %6.1f GB/s  gather(64K spans) %6.1f GB/s\n",
           name, bytes / r / 1e6, bytes / w / 1e6, gbytes / g / 1e6);
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], 0, 10) : 2032, bytes = mib << 20;
    CK(cudaFree(0));
    uint4 *sink; CK(cudaMalloc(&sink, 16));
    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror("open card0"); return 1; }
    struct drm_mode_create_dumb c = {0}; c.width = 4096; c.bpp = 32; c.height = bytes / (4096 * 4);
    if (ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c)) { perror("create dumb"); return 1; }
    struct drm_mode_map_dumb m = {0}; m.handle = c.handle;
    if (ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m)) { perror("map dumb"); return 1; }
    void *host = mmap(0, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    if (host == MAP_FAILED) { perror("mmap"); return 1; }

    // Path 2: export the dumb buffer as a dma-buf and import it into CUDA.
    {
        struct drm_prime_handle ph = {0}; ph.handle = c.handle; ph.flags = DRM_CLOEXEC | DRM_RDWR;
        if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph)) { perror("prime export"); }
        else {
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd = {}; hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
            hd.handle.fd = ph.fd; hd.size = c.size; hd.flags = 0;
            CUexternalMemory em; CUresult ir = cuImportExternalMemory(&em, &hd);
            printf("dma-buf import rc=%d\n", ir);
            if (ir == CUDA_SUCCESS) {
                CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = {}; bd.offset = 0; bd.size = c.size; CUdeviceptr p2 = 0;
                CUresult mr = cuExternalMemoryGetMappedBuffer(&p2, em, &bd);
                printf("dma-buf mapped rc=%d dev=%p\n", mr, (void *)p2);
                if (mr == CUDA_SUCCESS) report("dma-buf", (void *)p2, c.size, sink);
                cuDestroyExternalMemory(em);
            } else { const char *s; cuGetErrorString(ir, &s); printf("import error: %s\n", s); }
        }
    }
    CUresult rc = cuMemHostRegister(host, c.size, CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY);
    CUdeviceptr dev = 0; cuMemHostGetDevicePointer(&dev, host, 0);
    printf("carve-out %zu MiB registered rc=%d host=%p dev=%p uva=%s\n", (size_t)c.size >> 20, rc, host,
           (void *)dev, (uint64_t)host == dev ? "yes" : "no");
    if (rc) return 1;
    report("carve-out", (void *)dev, c.size, sink);
    void *ord; CK(cudaMalloc(&ord, 1ul << 30));
    report("cudaMalloc", ord, 1ul << 30, sink);
    cudaFree(ord);
    cuMemHostUnregister(host); munmap(host, c.size);
    struct drm_mode_destroy_dumb d = {c.handle}; ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    close(fd);
    return 0;
}
