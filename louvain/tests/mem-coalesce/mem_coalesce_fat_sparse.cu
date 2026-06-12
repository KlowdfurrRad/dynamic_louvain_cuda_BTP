// Microbenchmark: AoS vs SoA with a FAT record where the kernel uses only
// TWO fields that sit FAR APART in the struct (dest @ offset 4, w6 @ offset 48).
//
// This is AoS's worst case: the two fields land in different 32 B memory
// sectors, so AoS drags the whole 64 B record through the bus per edge while
// using just 12 B of it.  SoA reads only the two needed arrays, coalesced.
//
// Both layouts hold the SAME full 10-field data; only the access differs.
//
// Build:  ./compile.sh
// Run:    ./mem_coalesce_fat_sparse [n_edges]   (default 15M; ~2.0 GB total)

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

// --- AoS: same fat edge record, 10 fields, 64 bytes ---
struct FatEdge {
    int    src;        // offset  0
    int    dest;       // offset  4   <- used
    double weight;     // offset  8
    double w2;         // offset 16
    double w3;         // offset 24
    double w4;         // offset 32
    double w5;         // offset 40
    double w6;         // offset 48   <- used
    int    flags;      // offset 56
    int    id;         // offset 60
};                     // 64 bytes

__global__ void aos_kernel(const FatEdge* edges, double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = edges[i].dest + edges[i].w6;   // 2 far-apart fields of 64 B record
}

__global__ void soa_kernel(const int* dst, const double* w6, double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = dst[i] + w6[i];                // 2 coalesced array reads
}

int main(int argc, char** argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 15'000'000;
    const int threads = 256;
    const int blocks  = (m + threads - 1) / threads;
    const int reps    = 20;

    size_t aos_bytes = m * sizeof(FatEdge);
    size_t soa_bytes = m * (4ull * sizeof(int) + 6ull * sizeof(double));
    printf("edges = %d   sizeof(FatEdge) = %zu B   (only dest + w6 used)\n",
           m, sizeof(FatEdge));
    printf("AoS buffer %.1f MB, SoA buffers %.1f MB (same data, different layout)\n",
           aos_bytes / 1e6, soa_bytes / 1e6);

    // ---- host fill: every field gets a real value ----
    FatEdge* h_edges = (FatEdge*)malloc(aos_bytes);
    int*     h_dst   = (int*)malloc(m * sizeof(int));
    double*  h_w6    = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        int d = i % 1000;
        double w = 1.0 + (i % 7);
        h_edges[i] = { i, d, w, w + 1, w + 2, w + 3, w + 4, w + 5, i & 0xFF, i % 4096 };
        h_dst[i] = d;
        h_w6[i]  = w + 5;
    }

    // ---- device buffers: full 10-field data on BOTH sides ----
    FatEdge* d_edges; int* d_dst; double* d_w6; double* d_out;
    int* d_ints;      // src + flags + id           (3 unused int arrays)
    double* d_dbls;   // weight, w2..w5             (5 unused double arrays)
    CUDA_CHECK(cudaMalloc(&d_edges, aos_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst,  m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_w6,   m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ints, 3ull * m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dbls, 5ull * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,  m * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_edges, h_edges, aos_bytes,          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst,   h_dst,   m * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w6,    h_w6,    m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ints, 0, 3ull * m * sizeof(int)));     // never read
    CUDA_CHECK(cudaMemset(d_dbls, 0, 5ull * m * sizeof(double)));  // never read

    // ---- warm-up ----
    aos_kernel<<<blocks, threads>>>(d_edges, d_out, m);
    soa_kernel<<<blocks, threads>>>(d_dst, d_w6, d_out, m);
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
        soa_kernel<<<blocks, threads>>>(d_dst, d_w6, d_out, m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&soa_ms, start, stop));
    soa_ms /= reps;

    // effective bandwidth: bytes the layout forces through the bus + out write
    // (AoS: dest and w6 sit in different 32 B sectors -> whole 64 B record)
    double aos_gb = (m * (double)sizeof(FatEdge)                   + m * 8.0) / 1e9;
    double soa_gb = (m * (double)(sizeof(int) + sizeof(double))   + m * 8.0) / 1e9;

    printf("\nAoS: %8.3f ms   (%6.1f GB/s effective)\n", aos_ms, aos_gb / (aos_ms / 1e3));
    printf("SoA: %8.3f ms   (%6.1f GB/s effective)\n",   soa_ms, soa_gb / (soa_ms / 1e3));
    printf("\nSoA speed-up over AoS: %.2fx\n", aos_ms / soa_ms);

    cudaFree(d_edges); cudaFree(d_dst); cudaFree(d_w6);
    cudaFree(d_ints); cudaFree(d_dbls); cudaFree(d_out);
    free(h_edges); free(h_dst); free(h_w6);
    return 0;
}
