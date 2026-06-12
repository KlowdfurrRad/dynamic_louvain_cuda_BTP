// Microbenchmark: AoS vs SoA with a FAT record where the kernel uses ALL fields.
//
// This is AoS's best case: every byte of the 10-field record is consumed, so
// nothing is "wasted".  The remaining difference is purely access pattern --
// each AoS lane reads a 64 B record (warp accesses are 64 B-strided), while
// SoA reads 10 separate arrays, each one perfectly coalesced across the warp.
//
// Build:  ./compile.sh
// Run:    ./mem_coalesce_fat [n_edges]   (default 15M; ~2.0 GB total)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// --- AoS: a fat edge record, 10 fields ---
struct FatEdge {
    int    src;        // 4
    int    dest;       // 4
    double weight;     // 8
    double w2;         // 8
    double w3;         // 8
    double w4;         // 8
    double w5;         // 8
    double w6;         // 8
    int    flags;      // 4
    int    id;         // 4
};                     // 64 bytes

// processing uses every field, computed directly on the data in place
__global__ void aos_kernel(const FatEdge* edges, double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = edges[i].weight
           + 0.5     * edges[i].w2
           + 0.25    * edges[i].w3
           + 0.125   * edges[i].w4
           + 0.0625  * edges[i].w5
           + 0.03125 * edges[i].w6
           + (double)(edges[i].dest - edges[i].src) * 1e-6
           + (double)(edges[i].flags ^ edges[i].id) * 1e-9;
}

__global__ void soa_kernel(const int* src, const int* dst,
                           const double* w,  const double* w2, const double* w3,
                           const double* w4, const double* w5, const double* w6,
                           const int* flags, const int* id,
                           double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = w[i]
           + 0.5     * w2[i]
           + 0.25    * w3[i]
           + 0.125   * w4[i]
           + 0.0625  * w5[i]
           + 0.03125 * w6[i]
           + (double)(dst[i] - src[i]) * 1e-6
           + (double)(flags[i] ^ id[i]) * 1e-9;
}

int main(int argc, char** argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 15'000'000;
    const int threads = 256;
    const int blocks  = (m + threads - 1) / threads;
    const int reps    = 20;

    size_t aos_bytes = m * sizeof(FatEdge);
    size_t soa_bytes = m * (4ull * sizeof(int) + 6ull * sizeof(double));
    printf("edges = %d   sizeof(FatEdge) = %zu B   (all 10 fields used)\n",
           m, sizeof(FatEdge));
    printf("AoS buffer %.1f MB, SoA buffers %.1f MB\n",
           aos_bytes / 1e6, soa_bytes / 1e6);

    // ---- host fill: every field gets a real value ----
    FatEdge* h_edges = (FatEdge*)malloc(aos_bytes);
    int    *h_src = (int*)malloc(m * 4),    *h_dst   = (int*)malloc(m * 4);
    int    *h_fl  = (int*)malloc(m * 4),    *h_id    = (int*)malloc(m * 4);
    double *h_w   = (double*)malloc(m * 8), *h_w2 = (double*)malloc(m * 8);
    double *h_w3  = (double*)malloc(m * 8), *h_w4 = (double*)malloc(m * 8);
    double *h_w5  = (double*)malloc(m * 8), *h_w6 = (double*)malloc(m * 8);
    for (int i = 0; i < m; i++) {
        int s = i, d = i % 1000, fl = i & 0xFF, id = i % 4096;
        double w = 1.0 + (i % 7);
        h_edges[i] = { s, d, w, w + 1, w + 2, w + 3, w + 4, w + 5, fl, id };
        h_src[i] = s;  h_dst[i] = d;  h_fl[i] = fl;  h_id[i] = id;
        h_w[i] = w;     h_w2[i] = w + 1; h_w3[i] = w + 2;
        h_w4[i] = w + 3; h_w5[i] = w + 4; h_w6[i] = w + 5;
    }

    // ---- device buffers ----
    FatEdge* d_edges;
    int *d_src, *d_dst, *d_fl, *d_id;
    double *d_w, *d_w2, *d_w3, *d_w4, *d_w5, *d_w6, *d_out;
    CUDA_CHECK(cudaMalloc(&d_edges, aos_bytes));
    CUDA_CHECK(cudaMalloc(&d_src, m * 4));  CUDA_CHECK(cudaMalloc(&d_dst, m * 4));
    CUDA_CHECK(cudaMalloc(&d_fl,  m * 4));  CUDA_CHECK(cudaMalloc(&d_id,  m * 4));
    CUDA_CHECK(cudaMalloc(&d_w,  m * 8));   CUDA_CHECK(cudaMalloc(&d_w2, m * 8));
    CUDA_CHECK(cudaMalloc(&d_w3, m * 8));   CUDA_CHECK(cudaMalloc(&d_w4, m * 8));
    CUDA_CHECK(cudaMalloc(&d_w5, m * 8));   CUDA_CHECK(cudaMalloc(&d_w6, m * 8));
    CUDA_CHECK(cudaMalloc(&d_out, m * 8));
    CUDA_CHECK(cudaMemcpy(d_edges, h_edges, aos_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, m * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, h_dst, m * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fl,  h_fl,  m * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_id,  h_id,  m * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,  h_w,  m * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, m * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3, m * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4, h_w4, m * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5, h_w5, m * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w6, h_w6, m * 8, cudaMemcpyHostToDevice));

    // ---- warm-up ----
    aos_kernel<<<blocks, threads>>>(d_edges, d_out, m);
    soa_kernel<<<blocks, threads>>>(d_src, d_dst, d_w, d_w2, d_w3, d_w4,
                                    d_w5, d_w6, d_fl, d_id, d_out, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- timing ----
    cudaEvent_t start, stop;
    float aos_ms, soa_ms;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        aos_kernel<<<blocks, threads>>>(d_edges, d_out, m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&aos_ms, start, stop));
    aos_ms /= reps;

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        soa_kernel<<<blocks, threads>>>(d_src, d_dst, d_w, d_w2, d_w3, d_w4,
                                        d_w5, d_w6, d_fl, d_id, d_out, m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&soa_ms, start, stop));
    soa_ms /= reps;

    // both touch the same bytes: full record + out write
    double gb = (m * (double)sizeof(FatEdge) + m * 8.0) / 1e9;
    printf("\nAoS: %8.3f ms   (%6.1f GB/s)\n", aos_ms, gb / (aos_ms / 1e3));
    printf("SoA: %8.3f ms   (%6.1f GB/s)\n",   soa_ms, gb / (soa_ms / 1e3));
    printf("\nSoA speed-up over AoS: %.2fx\n", aos_ms / soa_ms);

    cudaFree(d_edges);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_fl); cudaFree(d_id);
    cudaFree(d_w); cudaFree(d_w2); cudaFree(d_w3); cudaFree(d_w4);
    cudaFree(d_w5); cudaFree(d_w6); cudaFree(d_out);
    free(h_edges);
    free(h_src); free(h_dst); free(h_fl); free(h_id);
    free(h_w); free(h_w2); free(h_w3); free(h_w4); free(h_w5); free(h_w6);
    return 0;
}
