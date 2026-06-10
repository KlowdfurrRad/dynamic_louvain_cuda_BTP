// ============================================================================
// df_louvain.cu  —  GPU Louvain (Static) + Dynamic Frontier (DF) + Delta-
//                   Screening (DS) + Naive-Dynamic (ND), written from scratch.
//
// Algorithms follow:
//   - S. Sahu, "DF Louvain: Fast Incrementally Expanding Approach for Community
//     Detection on Dynamic Graphs", arXiv:2404.19634v4  (Alg 1,2,3,4,5,6,7).
//   - N. Zarayeneh, A. Kalyanaraman, "Delta-Screening ...", IEEE TNSE 2021
//     (Alg 2 additions / Alg 3 deletions, Theorems 5 & 8).
//   - S. Sahu, "GVE-Louvain", arXiv:2312.04876 (the static base, ref [56]).
//
// Modularity (Eq.1) and delta-modularity (Eq.2), with M = total *directed*
// edge weight = 2*m (each undirected edge stored as two arcs):
//     Q              = sum_c [ sigma_c/M - (Sigma_c/M)^2 ]
//     dQ_{i: d->c}   = 2*(K_ic - K_id)/M - 2*k_i*(k_i + Sigma_c - Sigma_d)/M^2
//   where  K_ic = weight from i to community c (excluding self-loop),
//          k_i  = weighted degree of i,
//          Sigma_c = sum of weighted degrees of vertices in c (Sigma_d includes i),
//          sigma_c = weight of arcs internal to c (self-loop weight after aggregation).
//
// Design choices (deliberately different from the prior code, for correctness):
//   * Vertex-parallel, lock-free, ASYNCHRONOUS local moving (one thread/vertex),
//     best-neighbor-community selection. Community totals Sigma are maintained
//     with atomics, so the resolution/penalty term is ALWAYS correct (this is
//     the structural fix for the previous "k_i read from per-community array"
//     collapse).
//   * Per-vertex open-addressing hashtable in global memory (community -> weight)
//     sized to the vertex degree, so high-degree hubs never overflow.
//   * One kernel launch per local-moving iteration (no cooperative kernels):
//     simpler, portable, and the host can read the per-iteration delta-Q to test
//     convergence against a tolerance tau.
//   * Vertex pruning (process a vertex; on a move re-activate its neighbours) for
//     all variants; DF additionally GROWS the affected range on a move (frontier),
//     DS keeps a FIXED (larger) range, ND/Static put every vertex in range.
//   * Aggregation via Thrust sort_by_key + reduce_by_key on (src,dst) keys.
//
// I/O (compatible with real_graphs/.../run_benchmarks.sh):
//     n m
//     u v            x m            (initial undirected edges; weight assumed 1)
//     n_batches
//     n_del n_ins
//     u v            x n_del        (deletions)
//     u v            x n_ins        (insertions)
//     ... repeated per batch
//   argv[1] (optional) = file to write final community assignment of each method.
//
// Compile: nvcc -O3 -rdc=true -arch=sm_86 df_louvain.cu -o df_louvain
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <tuple>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Louvain hyper-parameters (match the DF paper, Section 5.1.2).
static const double TOLERANCE        = 1e-2;   // initial per-iteration tolerance tau
static const double TOLERANCE_DROP   = 10.0;   // tau shrinks by this each pass
static const double AGG_TOLERANCE    = 0.8;    // stop if communities shrink < this ratio
static const int    MAX_ITERATIONS   = 20;     // local-moving iterations per pass
static const int    MAX_PASSES       = 10;     // aggregation passes
#define MOVE_EPS 1e-12  // accept a move only if dQ > this

static const int    TPB = 256;                 // threads per block
static inline int   nblocks(int n) { return (n + TPB - 1) / TPB; }

// ============================================================================
// DEVICE HELPERS
// ============================================================================

__device__ __forceinline__ unsigned hash_u(int c) {
    return (unsigned)c * 2654435761u;          // Knuth multiplicative hash
}

// ============================================================================
// KERNELS
// ============================================================================

// K[v] = sum of weights of v's incident arcs (its weighted degree), read from CSR.
__global__ void computeK_kernel(int N, const int* off, const double* w, double* K) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    double s = 0.0;
    for (int e = off[v]; e < off[v + 1]; ++e) s += w[e];
    K[v] = s;
}

// Hashtable capacity per vertex = next power of two >= degree+1 (guarantees an
// empty slot so linear probing always terminates).
__global__ void htCap_kernel(int N, const int* off, int* cap) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int deg = off[v + 1] - off[v];
    int need = deg + 1;
    int c = 1;
    while (c < need) c <<= 1;
    cap[v] = c;
}

// Initialise every vertex to its own community: C[v] = v.
__global__ void identity_kernel(int N, int* C) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) C[v] = v;
}

__global__ void fill_int_kernel(int N, int* a, int val) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) a[v] = val;
}

// Local-moving iteration (Alg 5). One thread per vertex.
//   active[v] : vertex scheduled for processing this iteration (consumed).
//   range[v]  : vertex allowed to be processed at all (nullptr => all allowed).
//   isDF      : if true, a moved vertex's neighbours are added to the range.
//
// Correctness (this is the fix for the sub-singleton / stale-Sigma bug): a
// vertex picks its best target community OPTIMISTICALLY (against possibly-stale
// Sigma), then COMMITS only under a lock on the (old,new) community pair, where
// it re-reads Sigma and re-scans K_{v->old}/K_{v->new}. While that pair is
// locked, no other vertex can change the membership of either community (such a
// move would need the same locks), so the re-computed gain is EXACT and every
// committed move genuinely increases modularity — no stale-Sigma overshoot and
// no 2-cycle swap. C[v] is written only by v's own thread, so no per-vertex
// lock is needed.
__global__ void localMove_kernel(
        int N, const int* off, const int* dst, const double* w,
        int* C, const double* K, double* Sigma,
        int* active, int* range, int isDF,
        const long long* htOff, int* htKey, double* htVal,
        double M, double* dQ_accum, int* commLock)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    if (active[v] == 0) return;
    if (range != nullptr && range[v] == 0) { active[v] = 0; return; }
    active[v] = 0;                                  // mark processed

    long long base = htOff[v];
    int cap = (int)(htOff[v + 1] - htOff[v]);       // power of two
    int mask = cap - 1;

    // Clear this vertex's hashtable slice.
    for (int s = 0; s < cap; ++s) htKey[base + s] = -1;

    int d  = C[v];
    double ki = K[v];

    // Accumulate weight from v to each neighbouring community (skip self-loops).
    for (int e = off[v]; e < off[v + 1]; ++e) {
        int u = dst[e];
        if (u == v) continue;
        int c = C[u];
        double we = w[e];
        int slot = (int)(hash_u(c) & mask);
        while (true) {
            int k = htKey[base + slot];
            if (k == c)      { htVal[base + slot] += we; break; }
            if (k == -1)     { htKey[base + slot] = c; htVal[base + slot] = we; break; }
            slot = (slot + 1) & mask;
        }
    }

    // Optimistic K_{v->d} and best target (against possibly-stale Sigma).
    double Kid = 0.0;
    {
        int slot = (int)(hash_u(d) & mask);
        while (true) {
            int k = htKey[base + slot];
            if (k == d)  { Kid = htVal[base + slot]; break; }
            if (k == -1) break;
            slot = (slot + 1) & mask;
        }
    }
    double Sd = Sigma[d];
    int    best_c  = d;
    double best_dQ = 0.0;
    for (int s = 0; s < cap; ++s) {
        int c = htKey[base + s];
        if (c == -1 || c == d) continue;
        double Kic = htVal[base + s];
        double Sc  = Sigma[c];
        double dQ  = 2.0 * (Kic - Kid) / M
                   - 2.0 * ki * (ki + Sc - Sd) / (M * M);
        // strictly better, ties broken towards the smaller community id
        if (dQ > best_dQ + MOVE_EPS ||
            (dQ > best_dQ - MOVE_EPS && c < best_c)) {
            best_dQ = dQ;
            best_c  = c;
        }
    }
    if (best_c == d || best_dQ <= MOVE_EPS) return;

    // ---- verified commit under a lock on the (d, best_c) community pair ----
    int lo = min(d, best_c), hi = max(d, best_c);
    while (atomicCAS(&commLock[lo], 0, 1) != 0) {}
    while (atomicCAS(&commLock[hi], 0, 1) != 0) {}

    // Re-scan K_{v->d} and K_{v->best_c} with current memberships; while both
    // communities are locked their membership is frozen, so these and the Sigma
    // reads below are exact.
    double Kid2 = 0.0, Kic2 = 0.0;
    for (int e = off[v]; e < off[v + 1]; ++e) {
        int u = dst[e];
        if (u == v) continue;
        int cu = C[u];
        double we = w[e];
        if (cu == d)           Kid2 += we;
        else if (cu == best_c) Kic2 += we;
    }
    double Sd2 = Sigma[d], Sc2 = Sigma[best_c];
    double dQ2 = 2.0 * (Kic2 - Kid2) / M
               - 2.0 * ki * (ki + Sc2 - Sd2) / (M * M);

    if (dQ2 > MOVE_EPS) {
        Sigma[d]      -= ki;       // only the two lock holders touch these
        Sigma[best_c] += ki;
        C[v] = best_c;
        atomicAdd(dQ_accum, dQ2);
        __threadfence();           // publish C/Sigma before releasing the locks
        // Re-activate neighbours (and, for DF, expand the affected range).
        for (int e = off[v]; e < off[v + 1]; ++e) {
            int u = dst[e];
            active[u] = 1;
            if (isDF && range != nullptr) range[u] = 1;
        }
    }
    __threadfence();
    atomicExch(&commLock[hi], 0);
    atomicExch(&commLock[lo], 0);
}

// Mark community c as "present" (used before renumbering).
__global__ void markPresent_kernel(int N, const int* C, int* present) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) present[C[v]] = 1;
}

// Renumber: C[v] <- newid[C[v]] where newid = inclusiveScan(present)-1.
__global__ void renumber_kernel(int N, int* C, const int* newid) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) C[v] = newid[C[v]] - 1;
}

// Compose the dendrogram: orig[v] <- newid[ C[ orig[v] ] ] - 1.
// orig[v] is v's current super-vertex; C maps it to its (renumbered) community.
__global__ void fold_kernel(int Norig, int* orig, const int* C, const int* newid) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= Norig) return;
    orig[v] = newid[C[orig[v]]] - 1;
}

// Write src[e] = v for every arc e in v's adjacency (needed to form (src,dst) keys).
__global__ void writeSrc_kernel(int N, const int* off, int* src) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    for (int e = off[v]; e < off[v + 1]; ++e) src[e] = v;
}

// key[e] = (long long)C[src[e]] * nc + C[dst[e]]   (unique per (community,community)).
__global__ void buildKey_kernel(int M, const int* src, const int* dst,
                                const int* C, long long nc, long long* key) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;
    long long a = C[src[e]];
    long long b = C[dst[e]];
    key[e] = a * nc + b;
}

// Decode reduced keys back into (src,dst) and bump per-src arc counts.
__global__ void decode_kernel(int Mr, const long long* key, long long nc,
                              int* src, int* dst, int* cnt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Mr) return;
    long long k = key[e];
    int s = (int)(k / nc);
    int d = (int)(k % nc);
    src[e] = s;
    dst[e] = d;
    atomicAdd(&cnt[s], 1);
}

// ============================================================================
// DEVICE GRAPH (CSR of arcs) + small RAII-ish helpers
// ============================================================================

struct DGraph {
    int N = 0;        // vertices
    int M = 0;        // arcs (directed)
    int*    off = nullptr;   // size N+1
    int*    dst = nullptr;   // size M
    double* w   = nullptr;   // size M
};

static void freeDGraph(DGraph& g) {
    if (g.off) cudaFree(g.off);
    if (g.dst) cudaFree(g.dst);
    if (g.w)   cudaFree(g.w);
    g.off = nullptr; g.dst = nullptr; g.w = nullptr;
    g.N = g.M = 0;
}

// Build hashtable offsets (exclusive scan of per-vertex capacities) for graph g.
// Returns total capacity; fills d_htOff (size N+1).
static long long buildHashOffsets(const DGraph& g, long long* d_htOff, int* d_capScratch) {
    htCap_kernel<<<nblocks(g.N), TPB>>>(g.N, g.off, d_capScratch);
    CUDA_CHECK(cudaGetLastError());
    // 64-bit exclusive scan of per-vertex capacities into d_htOff (size N+1).
    thrust::device_ptr<int> cap(d_capScratch);
    thrust::device_vector<long long> tmp(g.N + 1);
    thrust::copy(cap, cap + g.N, tmp.begin());
    tmp[g.N] = 0;
    thrust::exclusive_scan(tmp.begin(), tmp.end(), thrust::device_ptr<long long>(d_htOff));
    long long total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_htOff + g.N, sizeof(long long), cudaMemcpyDeviceToHost));
    return total;
}

// Aggregate graph g by (renumbered) communities C into super-graph out (nc nodes).
// C must already be renumbered to [0, nc).  Self-loops are kept (they carry the
// internal community weight needed for correct degrees/modularity).
static void aggregate(const DGraph& g, const int* d_C, int nc, DGraph& out) {
    int M = g.M;

    int* d_src;  CUDA_CHECK(cudaMalloc(&d_src, sizeof(int) * M));
    writeSrc_kernel<<<nblocks(g.N), TPB>>>(g.N, g.off, d_src);
    CUDA_CHECK(cudaGetLastError());

    long long* d_key; CUDA_CHECK(cudaMalloc(&d_key, sizeof(long long) * M));
    buildKey_kernel<<<nblocks(M), TPB>>>(M, d_src, g.dst, d_C, (long long)nc, d_key);
    CUDA_CHECK(cudaGetLastError());

    // Sort arcs by (src,dst) key, carrying weights, then combine duplicates.
    double* d_w2; CUDA_CHECK(cudaMalloc(&d_w2, sizeof(double) * M));
    CUDA_CHECK(cudaMemcpy(d_w2, g.w, sizeof(double) * M, cudaMemcpyDeviceToDevice));
    thrust::device_ptr<long long> key(d_key);
    thrust::device_ptr<double>    wt(d_w2);
    thrust::sort_by_key(key, key + M, wt);

    long long* d_okey; CUDA_CHECK(cudaMalloc(&d_okey, sizeof(long long) * M));
    double*    d_ow;   CUDA_CHECK(cudaMalloc(&d_ow,   sizeof(double)    * M));
    auto end = thrust::reduce_by_key(key, key + M, wt,
                                     thrust::device_ptr<long long>(d_okey),
                                     thrust::device_ptr<double>(d_ow));
    int Mr = (int)(end.first - thrust::device_ptr<long long>(d_okey));

    // Decode reduced keys to (src,dst) and count arcs per super-vertex.
    out.N = nc;
    out.M = Mr;
    CUDA_CHECK(cudaMalloc(&out.dst, sizeof(int)    * Mr));
    CUDA_CHECK(cudaMalloc(&out.w,   sizeof(double) * Mr));
    CUDA_CHECK(cudaMalloc(&out.off, sizeof(int)    * (nc + 1)));
    CUDA_CHECK(cudaMemset(out.off, 0, sizeof(int) * (nc + 1)));
    CUDA_CHECK(cudaMemcpy(out.w, d_ow, sizeof(double) * Mr, cudaMemcpyDeviceToDevice));

    int* d_osrc; CUDA_CHECK(cudaMalloc(&d_osrc, sizeof(int) * Mr));
    decode_kernel<<<nblocks(Mr), TPB>>>(Mr, d_okey, (long long)nc, d_osrc, out.dst, out.off);
    CUDA_CHECK(cudaGetLastError());

    // offsets = exclusive scan of per-src counts (counts currently live in out.off[0..nc)).
    thrust::device_ptr<int> offp(out.off);
    thrust::exclusive_scan(offp, offp + nc + 1, offp);   // out.off[nc] becomes total = Mr

    cudaFree(d_src); cudaFree(d_key); cudaFree(d_w2);
    cudaFree(d_okey); cudaFree(d_ow); cudaFree(d_osrc);
}

// Count distinct communities present in C (size N).
static int countCommunities(const int* d_C, int N, int* d_present /*size>=N*/) {
    CUDA_CHECK(cudaMemset(d_present, 0, sizeof(int) * N));
    markPresent_kernel<<<nblocks(N), TPB>>>(N, d_C, d_present);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<int> p(d_present);
    return (int)thrust::reduce(p, p + N);
}

// ============================================================================
// CORE LOUVAIN (Alg 4)  — shared by Static / ND / DF / DS
//   Inputs are host CSR of the *current full graph*, the warm-start community of
//   each vertex, and the initial active/range masks for pass 0 (frontier).
//   Returns the final community of every original vertex (the dendrogram top).
// ============================================================================

struct LouvainOut {
    vector<int> comm;   // final community per original vertex
    int passes = 0;
};

static LouvainOut louvain(
        int N,
        const vector<int>&    h_off,
        const vector<int>&    h_dst,
        const vector<double>& h_w,
        const vector<int>&    h_initComm,    // warm start (identity for Static)
        const vector<char>&   h_active0,     // pass-0 active mask
        const vector<char>&   h_range0,      // pass-0 range mask
        bool isDF)
{
    int M = (int)h_dst.size();

    // ---- upload current full graph (this is "G'" at pass 0) ----
    DGraph g;
    g.N = N; g.M = M;
    CUDA_CHECK(cudaMalloc(&g.off, sizeof(int) * (N + 1)));
    CUDA_CHECK(cudaMalloc(&g.dst, sizeof(int) * M));
    CUDA_CHECK(cudaMalloc(&g.w,   sizeof(double) * M));
    CUDA_CHECK(cudaMemcpy(g.off, h_off.data(), sizeof(int) * (N + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.dst, h_dst.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.w,   h_w.data(),   sizeof(double) * M, cudaMemcpyHostToDevice));

    // ---- per-vertex degree K (constant for a given graph) ----
    double* d_K; CUDA_CHECK(cudaMalloc(&d_K, sizeof(double) * N));
    computeK_kernel<<<nblocks(N), TPB>>>(N, g.off, g.w, d_K);
    CUDA_CHECK(cudaGetLastError());

    // total directed weight M_total = sum_v K[v]
    double M_total = thrust::reduce(thrust::device_ptr<double>(d_K),
                                    thrust::device_ptr<double>(d_K) + N, 0.0);

    // ---- community of each current vertex (warm start) ----
    int* d_C; CUDA_CHECK(cudaMalloc(&d_C, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(d_C, h_initComm.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

    // ---- Sigma[c] = sum of K over vertices currently in community c ----
    double* d_Sigma; CUDA_CHECK(cudaMalloc(&d_Sigma, sizeof(double) * N));
    {
        // Sigma computed on host from warm start + K (correct initial totals).
        vector<double> hK(N);
        CUDA_CHECK(cudaMemcpy(hK.data(), d_K, sizeof(double) * N, cudaMemcpyDeviceToHost));
        vector<double> hSigma(N, 0.0);
        for (int v = 0; v < N; ++v) hSigma[h_initComm[v]] += hK[v];
        CUDA_CHECK(cudaMemcpy(d_Sigma, hSigma.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    }

    // ---- dendrogram: orig[v] = current super-vertex of original vertex v ----
    int* d_orig; CUDA_CHECK(cudaMalloc(&d_orig, sizeof(int) * N));
    identity_kernel<<<nblocks(N), TPB>>>(N, d_orig);
    CUDA_CHECK(cudaGetLastError());

    // ---- active / range masks (pass 0 supplied; later passes = all) ----
    int* d_active; CUDA_CHECK(cudaMalloc(&d_active, sizeof(int) * N));
    int* d_range;  CUDA_CHECK(cudaMalloc(&d_range,  sizeof(int) * N));
    {
        vector<int> ia(N), ir(N);
        for (int v = 0; v < N; ++v) { ia[v] = h_active0[v]; ir[v] = h_range0[v]; }
        CUDA_CHECK(cudaMemcpy(d_active, ia.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_range,  ir.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    }

    // ---- scratch ----
    int* d_present; CUDA_CHECK(cudaMalloc(&d_present, sizeof(int) * N));
    int* d_cap;     CUDA_CHECK(cudaMalloc(&d_cap,     sizeof(int) * N));
    long long* d_htOff; CUDA_CHECK(cudaMalloc(&d_htOff, sizeof(long long) * (N + 1)));
    double* d_dQ;   CUDA_CHECK(cudaMalloc(&d_dQ, sizeof(double)));
    // Per-community lock (indexed by community id, < g.N <= N). Used to make each
    // local-move commit verify its gain against frozen Sigma/memberships.
    int* d_commLock; CUDA_CHECK(cudaMalloc(&d_commLock, sizeof(int) * N));

    // hashtable storage sized to pass-0 total capacity (later passes are smaller)
    long long htTotal = buildHashOffsets(g, d_htOff, d_cap);
    int*    d_htKey; CUDA_CHECK(cudaMalloc(&d_htKey, sizeof(int)    * htTotal));
    double* d_htVal; CUDA_CHECK(cudaMalloc(&d_htVal, sizeof(double) * htTotal));
    long long htCapAlloc = htTotal;

    double tau = TOLERANCE;
    int passes = 0;
    bool firstPass = true;

    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        passes++;

        // ----- local-moving phase (Alg 5) -----
        int*  active_ptr = d_active;
        int*  range_ptr  = (firstPass ? d_range : nullptr);  // later passes: no range limit
        int   df_flag    = (firstPass && isDF) ? 1 : 0;

        CUDA_CHECK(cudaMemset(d_commLock, 0, sizeof(int) * g.N));  // all locks free

        int iters = 0;
        for (int it = 0; it < MAX_ITERATIONS; ++it) {
            iters++;
            CUDA_CHECK(cudaMemset(d_dQ, 0, sizeof(double)));
            localMove_kernel<<<nblocks(g.N), TPB>>>(
                g.N, g.off, g.dst, g.w, d_C, d_K, d_Sigma,
                active_ptr, range_ptr, df_flag,
                d_htOff, d_htKey, d_htVal, M_total, d_dQ, d_commLock);
            CUDA_CHECK(cudaGetLastError());
            double hdQ = 0.0;
            CUDA_CHECK(cudaMemcpy(&hdQ, d_dQ, sizeof(double), cudaMemcpyDeviceToHost));
            if (hdQ <= tau) break;     // locally converged
        }

        // ----- renumber communities & fold dendrogram -----
        int nc = countCommunities(d_C, g.N, d_present);
        // newid = inclusive scan of present  (present already in d_present)
        thrust::device_ptr<int> pp(d_present);
        thrust::inclusive_scan(pp, pp + g.N, pp);
        fold_kernel<<<nblocks(N), TPB>>>(N, d_orig, d_C, d_present);
        CUDA_CHECK(cudaGetLastError());
        renumber_kernel<<<nblocks(g.N), TPB>>>(g.N, d_C, d_present);
        CUDA_CHECK(cudaGetLastError());

        // ----- convergence checks (Alg 4 lines 11-13) -----
        bool converged = (iters <= 1);                       // no productive moves
        bool lowShrink = ((double)nc / (double)g.N) > AGG_TOLERANCE;
        if (converged || lowShrink || nc == g.N) break;

        // ----- aggregation phase (Alg 6) -----
        DGraph ng;
        aggregate(g, d_C, nc, ng);
        freeDGraph(g);
        g = ng;

        // new K = super-vertex degrees; new Sigma = K (each super-vertex alone);
        // new C = identity; all super-vertices active and in range.
        cudaFree(d_K);     CUDA_CHECK(cudaMalloc(&d_K,     sizeof(double) * g.N));
        cudaFree(d_Sigma); CUDA_CHECK(cudaMalloc(&d_Sigma, sizeof(double) * g.N));
        computeK_kernel<<<nblocks(g.N), TPB>>>(g.N, g.off, g.w, d_K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_Sigma, d_K, sizeof(double) * g.N, cudaMemcpyDeviceToDevice));
        identity_kernel<<<nblocks(g.N), TPB>>>(g.N, d_C);
        CUDA_CHECK(cudaGetLastError());
        fill_int_kernel<<<nblocks(g.N), TPB>>>(g.N, d_active, 1);
        CUDA_CHECK(cudaGetLastError());

        // rebuild hashtable offsets for the (smaller) aggregated graph
        long long t = buildHashOffsets(g, d_htOff, d_cap);
        if (t > htCapAlloc) {   // should not happen (graph shrinks), but be safe
            cudaFree(d_htKey); cudaFree(d_htVal);
            CUDA_CHECK(cudaMalloc(&d_htKey, sizeof(int)    * t));
            CUDA_CHECK(cudaMalloc(&d_htVal, sizeof(double) * t));
            htCapAlloc = t;
        }

        tau /= TOLERANCE_DROP;
        firstPass = false;
    }

    // ---- download final community of each original vertex ----
    LouvainOut res;
    res.comm.resize(N);
    res.passes = passes;
    CUDA_CHECK(cudaMemcpy(res.comm.data(), d_orig, sizeof(int) * N, cudaMemcpyDeviceToHost));

    freeDGraph(g);
    cudaFree(d_K); cudaFree(d_C); cudaFree(d_Sigma); cudaFree(d_orig);
    cudaFree(d_active); cudaFree(d_range);
    cudaFree(d_present); cudaFree(d_cap); cudaFree(d_htOff); cudaFree(d_dQ);
    cudaFree(d_commLock);
    cudaFree(d_htKey); cudaFree(d_htVal);
    return res;
}

// ============================================================================
// HOST UTILITIES
// ============================================================================

// Build CSR (arc form) from an adjacency list; also returns total arc count.
static void buildCSR(const vector<vector<pair<int,double>>>& adj,
                     vector<int>& off, vector<int>& dst, vector<double>& w) {
    int N = (int)adj.size();
    off.assign(N + 1, 0);
    for (int v = 0; v < N; ++v) off[v + 1] = off[v] + (int)adj[v].size();
    int M = off[N];
    dst.resize(M);
    w.resize(M);
    for (int v = 0; v < N; ++v) {
        int p = off[v];
        for (auto& e : adj[v]) { dst[p] = e.first; w[p] = e.second; ++p; }
    }
}

// True modularity Q of partition `comm` on the graph given by CSR.
//   Q = sum_c [ sigma_c/M - (Sigma_c/M)^2 ],  M = total arc weight.
static double modularity(const vector<int>& off, const vector<int>& dst,
                         const vector<double>& w, const vector<int>& comm) {
    int N = (int)off.size() - 1;
    double M = 0.0;
    for (double x : w) M += x;
    if (M == 0.0) return 0.0;
    map<int,double> sigma, internal;
    for (int v = 0; v < N; ++v) {
        int cv = comm[v];
        for (int e = off[v]; e < off[v + 1]; ++e) {
            sigma[cv] += w[e];
            if (comm[dst[e]] == cv) internal[cv] += w[e];
        }
    }
    double Q = 0.0;
    for (auto& kv : sigma) {
        double Sc = kv.second;
        double in = internal.count(kv.first) ? internal[kv.first] : 0.0;
        Q += in / M - (Sc / M) * (Sc / M);
    }
    return Q;
}

static int numCommunities(const vector<int>& comm) {
    set<int> s(comm.begin(), comm.end());
    return (int)s.size();
}

// ---- marking: build (active,range) masks per variant -------------------------
// All use the previous community `prevC`, the *new* graph CSR (off,dst), and the
// batch (deletions/insertions). prevK/prevSigma/prevM are auxiliary info from the
// previous snapshot (used by DS's modularity screening).

// ND (Alg 2): every vertex is affected and in range.
static void markND(int N, vector<char>& active, vector<char>& range) {
    active.assign(N, 1);
    range.assign(N, 1);
}

// DF (Alg 1, initial marking): endpoints of intra-community deletions and
// inter-community insertions. (Frontier growth happens later, on the GPU.)
static void markDF(int N, const vector<int>& prevC,
                   const vector<tuple<int,int>>& del,
                   const vector<tuple<int,int>>& ins,
                   vector<char>& active, vector<char>& range) {
    active.assign(N, 0);
    range.assign(N, 0);
    for (auto& e : del) {
        int i = get<0>(e), j = get<1>(e);
        if (prevC[i] == prevC[j]) { active[i] = range[i] = 1; active[j] = range[j] = 1; }
    }
    for (auto& e : ins) {
        int i = get<0>(e), j = get<1>(e);
        if (prevC[i] != prevC[j]) { active[i] = range[i] = 1; active[j] = range[j] = 1; }
    }
}

// DS: mark source vertices, their neighbours, and the *entire* affected
// community. Fixed range (no frontier growth).
//   deletions (i,j), same community  -> mark i, neighbours of i, community C[i]
//   insertions from i -> for the target community c* with the best modularity
//      gain among i's inter-community insertions: mark i, neighbours of i, comm c*
//
// NOTE: this is the PARALLEL delta-screening of the DF Louvain paper
// (Sahu, arXiv:2404.19634, Alg. 3), which simplifies the original sequential
// algorithm of Zarayeneh & Kalyanaraman (IEEE TNSE 2021, Alg. 2/3) in two ways:
//   (1) the insertion screening gain uses ONLY the newly-inserted edge weights
//       as K_{i->c} (Kto[c] below), not i's full edge weight to community c;
//   (2) the original's deferral check is dropped -- it marks from source i when
//       the best screening gain is positive, without requiring that gain_{i->c*}
//       >= gain_{c*-vertex -> C(i)} (i.e. without deferring to the endpoint with
//       the stronger incentive). Dropping it only enlarges the affected set, so
//       the marked set remains a safe superset of the original's.
static void markDS(int N, const vector<int>& prevC,
                   const vector<int>& off, const vector<int>& dst,
                   const vector<double>& prevK, const vector<double>& prevSigma, double prevM,
                   const vector<tuple<int,int>>& del,
                   const vector<tuple<int,int>>& ins,
                   vector<char>& active, vector<char>& range) {
    vector<char> markV(N, 0), markE(N, 0);     // vertex affected / its-neighbours affected
    vector<char> markC(N, 0);                  // community affected

    // deletions within a community
    for (auto& e : del) {
        int i = get<0>(e), j = get<1>(e);
        if (prevC[i] == prevC[j]) { markV[i] = 1; markE[i] = 1; markC[prevC[j]] = 1; }
    }

    // insertions: group by source, accumulate the INSERTED weight to each target
    // community (simplification (1) in the header), then pick the community c*
    // with the highest screening delta-modularity and mark if it is positive --
    // with no deferral to the opposite endpoint (simplification (2) in the header).
    // ins is sorted-by-source by the caller.
    size_t p = 0;
    while (p < ins.size()) {
        int i = get<0>(ins[p]);
        int d = prevC[i];
        double ki = prevK[i];
        map<int,double> Kto;       // community -> inserted weight from i
        size_t q = p;
        while (q < ins.size() && get<0>(ins[q]) == i) {
            int j = get<1>(ins[q]);
            int c = prevC[j];
            if (c != d) Kto[c] += 1.0;   // unit weight
            ++q;
        }
        p = q;
        int    bestC  = -1;
        double bestDQ = 0.0;
        for (auto& kv : Kto) {
            int c = kv.first;
            double Kic = kv.second;
            double Sc  = prevSigma[c];
            double Sd  = prevSigma[d];
            double dQ  = 2.0 * Kic / prevM - 2.0 * ki * (ki + Sc - Sd) / (prevM * prevM);
            if (dQ > bestDQ) { bestDQ = dQ; bestC = c; }
        }
        if (bestC >= 0) { markV[i] = 1; markE[i] = 1; markC[bestC] = 1; }
    }

    // propagation: neighbours of marked vertices, and whole marked communities
    active.assign(N, 0);
    range.assign(N, 0);
    for (int v = 0; v < N; ++v) {
        if (markE[v]) {
            for (int e = off[v]; e < off[v + 1]; ++e) markV[dst[e]] = 1;
        }
    }
    for (int v = 0; v < N; ++v) {
        if (markV[v] || markC[prevC[v]]) { active[v] = 1; range[v] = 1; }
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);

    int N, m;
    if (!(cin >> N >> m)) { fprintf(stderr, "bad input\n"); return 1; }

    vector<vector<pair<int,double>>> adj(N);
    for (int i = 0; i < m; ++i) {
        int u, v; cin >> u >> v;
        adj[u].emplace_back(v, 1.0);
        adj[v].emplace_back(u, 1.0);
    }

    // ---- build initial CSR ----
    vector<int> off, dst; vector<double> w;
    buildCSR(adj, off, dst, w);

    auto degK = [&](const vector<int>& o, const vector<double>& ww) {
        vector<double> K(N, 0.0);
        for (int v = 0; v < N; ++v) for (int e = o[v]; e < o[v + 1]; ++e) K[v] += ww[e];
        return K;
    };
    auto sigmaOf = [&](const vector<int>& comm, const vector<double>& K) {
        vector<double> S(N, 0.0);
        for (int v = 0; v < N; ++v) S[comm[v]] += K[v];
        return S;
    };
    auto totalW = [&](const vector<double>& ww) {
        double t = 0.0; for (double x : ww) t += x; return t;
    };

    // =================== STATIC LOUVAIN on the initial graph ===================
    vector<int>  identC(N);
    for (int v = 0; v < N; ++v) identC[v] = v;
    vector<char> allA(N, 1), allR(N, 1);

    auto t0 = chrono::steady_clock::now();
    LouvainOut st = louvain(N, off, dst, w, identC, allA, allR, /*isDF=*/false);
    auto t1 = chrono::steady_clock::now();
    long staticMs = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    double staticQ = modularity(off, dst, w, st.comm);
    printf("===== STATIC =====\n");
    printf("Static: Q=%.6f  communities=%d  passes=%d  time=%ld ms\n",
           staticQ, numCommunities(st.comm), st.passes, staticMs);

    // Per-method running state (community + auxiliary K/Sigma + total weight).
    struct State { vector<int> comm; vector<double> K, Sigma; double M; };
    auto makeState = [&](const vector<int>& comm) {
        State s; s.comm = comm; s.K = degK(off, w); s.Sigma = sigmaOf(comm, s.K);
        s.M = totalW(w); return s;
    };
    State sND = makeState(st.comm);
    State sDF = makeState(st.comm);
    State sDS = makeState(st.comm);

    const char* outFile = (argc > 1) ? argv[1] : nullptr;

    // ---- optional per-batch assignment dump (enable with env DF_DUMP=1) ----
    // Writes <prefix>_b<k>[_<method>].txt (one "v comm" per line) so the partition can
    // be compared across batches with compare_allocations.py. prefix is derived from
    // argv[1] (strip "_communities.txt" / ".txt"), organising dumps per graph.
    bool doDump = (getenv("DF_DUMP") != nullptr) && outFile;
    string dumpPrefix;
    if (doDump) {
        string s = outFile;
        const string suf = "_communities.txt";
        if (s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0)
            dumpPrefix = s.substr(0, s.size() - suf.size());
        else if (s.size() >= 4 && s.compare(s.size() - 4, 4, ".txt") == 0)
            dumpPrefix = s.substr(0, s.size() - 4);
        else
            dumpPrefix = s;
    }
    auto dumpComm = [&](const string& tag, const vector<int>& comm) {
        if (!doDump) return;
        ofstream f(dumpPrefix + "_" + tag + ".txt");
        for (int v = 0; v < N; ++v) f << v << ' ' << comm[v] << '\n';
    };
    dumpComm("b0", st.comm);     // shared static start for ND/DF/DS

    // =================== DYNAMIC BATCHES ===================
    int nBatches;
    if (cin >> nBatches) {
        for (int b = 0; b < nBatches; ++b) {
            int nDel, nIns; cin >> nDel >> nIns;
            vector<tuple<int,int>> del(nDel), ins(nIns);
            for (int i = 0; i < nDel; ++i) { int u, v; cin >> u >> v; del[i] = {u, v}; }
            for (int i = 0; i < nIns; ++i) { int u, v; cin >> u >> v; ins[i] = {u, v}; }

            // Mark affected on the PREVIOUS graph/communities, then apply update.
            // (DS needs the new-graph neighbours; we apply first, then mark — the
            //  initial endpoint test uses prevC which is unchanged by the update.)
            // Apply batch to adjacency:
            for (auto& e : del) {
                int u = get<0>(e), v = get<1>(e);
                auto rm = [&](int a, int b2){
                    auto& A = adj[a];
                    for (size_t k = 0; k < A.size(); ++k)
                        if (A[k].first == b2) { A[k] = A.back(); A.pop_back(); break; }
                };
                rm(u, v); rm(v, u);
            }
            for (auto& e : ins) {
                int u = get<0>(e), v = get<1>(e);
                adj[u].emplace_back(v, 1.0);
                adj[v].emplace_back(u, 1.0);
            }
            buildCSR(adj, off, dst, w);

            // duplicate the batch with reversed arcs so endpoint-marking covers
            // both i and j (the input batch is undirected but listed once).
            vector<tuple<int,int>> delB = del, insB = ins;
            for (auto& e : del) delB.emplace_back(get<1>(e), get<0>(e));
            for (auto& e : ins) insB.emplace_back(get<1>(e), get<0>(e));
            // DS expects insertions sorted by source vertex.
            vector<tuple<int,int>> insSorted = insB;
            sort(insSorted.begin(), insSorted.end(),
                 [](const tuple<int,int>& a, const tuple<int,int>& c){ return get<0>(a) < get<0>(c); });

            vector<double> Knew = degK(off, w);
            double Mnew = totalW(w);

            // ---------- ND ----------
            {
                vector<char> A, R; markND(N, A, R);
                auto t = chrono::steady_clock::now();
                LouvainOut r = louvain(N, off, dst, w, sND.comm, A, R, false);
                auto t2 = chrono::steady_clock::now();
                long ms = chrono::duration_cast<chrono::milliseconds>(t2 - t).count();
                double Q = modularity(off, dst, w, r.comm);
                printf("[batch %d] ND: Q=%.6f comms=%d time=%ld ms\n",
                       b + 1, Q, numCommunities(r.comm), ms);
                sND.comm = r.comm; sND.K = Knew; sND.Sigma = sigmaOf(r.comm, Knew); sND.M = Mnew;
                dumpComm("b" + to_string(b + 1) + "_ND", sND.comm);
            }
            // ---------- DF ----------
            {
                vector<char> A, R; markDF(N, sDF.comm, delB, insB, A, R);
                int naff = 0; for (char c : A) naff += c;
                auto t = chrono::steady_clock::now();
                LouvainOut r = louvain(N, off, dst, w, sDF.comm, A, R, /*isDF=*/true);
                auto t2 = chrono::steady_clock::now();
                long ms = chrono::duration_cast<chrono::milliseconds>(t2 - t).count();
                double Q = modularity(off, dst, w, r.comm);
                printf("[batch %d] DF: Q=%.6f comms=%d affected0=%d time=%ld ms\n",
                       b + 1, Q, numCommunities(r.comm), naff, ms);
                sDF.comm = r.comm; sDF.K = Knew; sDF.Sigma = sigmaOf(r.comm, Knew); sDF.M = Mnew;
                dumpComm("b" + to_string(b + 1) + "_DF", sDF.comm);
            }
            // ---------- DS ----------
            {
                vector<char> A, R;
                markDS(N, sDS.comm, off, dst, sDS.K, sDS.Sigma, sDS.M, delB, insSorted, A, R);
                int naff = 0; for (char c : A) naff += c;
                auto t = chrono::steady_clock::now();
                LouvainOut r = louvain(N, off, dst, w, sDS.comm, A, R, /*isDF=*/false);
                auto t2 = chrono::steady_clock::now();
                long ms = chrono::duration_cast<chrono::milliseconds>(t2 - t).count();
                double Q = modularity(off, dst, w, r.comm);
                printf("[batch %d] DS: Q=%.6f comms=%d affected0=%d time=%ld ms\n",
                       b + 1, Q, numCommunities(r.comm), naff, ms);
                sDS.comm = r.comm; sDS.K = Knew; sDS.Sigma = sigmaOf(r.comm, Knew); sDS.M = Mnew;
                dumpComm("b" + to_string(b + 1) + "_DS", sDS.comm);
            }
        }
    }

    // ---- write DF final communities (the recommended method) ----
    if (outFile) {
        ofstream fout(outFile);
        set<int> uniq(sDF.comm.begin(), sDF.comm.end());
        fout << uniq.size() << "\n";
        for (int v = 0; v < N; ++v) fout << v << " " << sDF.comm[v] << "\n";
        printf("DF communities written to %s\n", outFile);
    }
    return 0;
}
