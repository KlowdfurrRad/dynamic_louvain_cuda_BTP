// Simple microbenchmark: AoS vs SoA memory access on the GPU.
// Each thread reads one edge's (dest, weight) and accumulates a value,
// mirroring the edge-scan pattern in the Louvain kernels.
//
// Both layouts hold the SAME data -- src, dest, weight per edge (16 B/edge) --
// so the comparison is purely about layout.  The kernels only READ dest+weight.
//
// Build:  ./compile.sh
// Run:    ./mem_coalesce [n_edges]   (default 50M; ~2.0 GB total)

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

// --- AoS: same layout as the UGRC implementation ---
struct Edge {
    int src;
    int dest;
    double weight;
};  // 16 bytes

__global__ void aos_kernel(const Edge* edges, double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = edges[i].dest + edges[i].weight;   // strided 16B reads
}

// --- SoA: separate arrays, as in the present implementation ---
__global__ void soa_kernel(const int* dst, const double* w, double* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = dst[i] + w[i];                     // adjacent-word reads
}

int main(int argc, char** argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 50'000'000;   // edges
    const int threads = 256;
    const int blocks  = (m + threads - 1) / threads;
    const int reps    = 20;

    printf("edges = %d  (AoS buffer %.1f MB, SoA buffers %.1f MB)\n",
           m, m * sizeof(Edge) / 1e6,
           m * (2 * sizeof(int) + sizeof(double)) / 1e6);

    // host fill (values don't matter, just not constant)
    Edge*   h_edges = (Edge*)malloc(m * sizeof(Edge));
    int*    h_src   = (int*)malloc(m * sizeof(int));
    int*    h_dst   = (int*)malloc(m * sizeof(int));
    double* h_w     = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        h_edges[i] = { i, i % 1000, 1.0 + (i % 7) };
        h_src[i]   = i;
        h_dst[i]   = i % 1000;
        h_w[i]     = 1.0 + (i % 7);
    }

    Edge* d_edges; int* d_src; int* d_dst; double* d_w; double* d_out;
    CUDA_CHECK(cudaMalloc(&d_edges, m * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_src,   m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst,   m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_w,     m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,   m * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_edges, h_edges, m * sizeof(Edge),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src,   h_src,   m * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst,   h_dst,   m * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,     h_w,     m * sizeof(double), cudaMemcpyHostToDevice));

    // warm-up
    aos_kernel<<<blocks, threads>>>(d_edges, d_out, m);
    soa_kernel<<<blocks, threads>>>(d_dst, d_w, d_out, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
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
        soa_kernel<<<blocks, threads>>>(d_dst, d_w, d_out, m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&soa_ms, start, stop));
    soa_ms /= reps;

    // effective bandwidth: bytes the layout forces through the bus + out write
    double aos_gb = (m * (double)sizeof(Edge)                    + m * 8.0) / 1e9;
    double soa_gb = (m * (double)(sizeof(int) + sizeof(double))  + m * 8.0) / 1e9;

    printf("\nAoS: %8.3f ms   (%6.1f GB/s effective)\n", aos_ms, aos_gb / (aos_ms / 1e3));
    printf("SoA: %8.3f ms   (%6.1f GB/s effective)\n",   soa_ms, soa_gb / (soa_ms / 1e3));
    printf("\nSoA speed-up over AoS: %.2fx\n", aos_ms / soa_ms);

    cudaFree(d_edges); cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_out);
    free(h_edges); free(h_src); free(h_dst); free(h_w);
    return 0;
}
