// EXPERIMENTAL: Node-based variant of cuda dynamic louvain.
// Uses node-based "best community" selection instead of edge-based greedy.
// Each node scans all neighbors, picks the best community, moves once per iteration.
// NOTE: This produced worse results (0.518 vs 0.873 modularity on GermanyRoad)
//       because the edge-based approach allows multiple moves per node per iteration,
//       creating a cascading propagation effect that builds larger communities.
//       Kept for reference only.
//
// Dynamic Louvain uses previous community assignments as a starting point
// after graph updates (edge insertions/deletions), converging much faster
// than re-running static Louvain from scratch.
//
// Input format:
//   n_nodes m_edges
//   u1 v1 w1          (undirected edges)
//   ...
//   n_batches
//   n_deletions1 n_insertions1
//   del_u1 del_v1 del_w1
//   ...
//   ins_u1 ins_v1 ins_w1
//   ...
//   [repeat for each batch]
//
// Compile: nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
// Requires compute capability >= 6.0 (for double atomicAdd and cooperative groups)

#include <vector>
using namespace std;

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// #define DEBUG

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


// ============================================================
// STRUCTURES
// ============================================================

struct Edge {
    int src;
    int dest;
    double weight;
};

struct EdgeComparator {
    __host__ __device__
    bool operator()(const struct Edge &a, const struct Edge &b) const {
        if (a.src == b.src) return a.dest < b.dest;
        return a.src < b.src;
    }
};

struct GetKey {
    __host__ __device__
    thrust::pair<int,int> operator()(const Edge &e) const {
        return {e.src, e.dest};
    }
};

struct GetWeight {
    __host__ __device__
    double operator()(const Edge &e) const {
        return e.weight;
    }
};

struct StoreKey {
    __host__ __device__
    Edge operator()(const thrust::pair<int,int> &p) const {
        return Edge{p.first, p.second, 0.0};
    }
};

struct CountEdgesBySrc {
    int* offsets;
    CountEdgesBySrc(int* o) : offsets(o) {}
    __device__ void operator()(const Edge& e) const {
        atomicAdd(&offsets[e.src], 1);
    }
};

// State returned by Louvain algorithm, usable as input for dynamic runs
struct LouvainState {
    int n_nodes;                       // number of original vertices
    vector<int> community;             // community[v] for each original vertex
    vector<double> vertex_weight;      // total edge weight of each original vertex
    vector<double> community_weight;   // total edge weight of each community (indexed by community ID)
    double total_weight;               // sum of all directed edge weights
};


// ============================================================
// DEVICE FUNCTIONS
// ============================================================

__device__ double update_community_degree(
    int n_nodes, struct Edge *d_csr_adj, int *d_csr_node_offset,
    int *community, double *community_degree,
    int node_to_move, int old_community, int new_community)
{
    for (int i = d_csr_node_offset[node_to_move]; i < d_csr_node_offset[node_to_move + 1]; i++) {
        community_degree[old_community] -= d_csr_adj[i].weight;
        community_degree[new_community] += d_csr_adj[i].weight;
    }
    return 0.0;
}

__device__ double calculate_modularity_change(
    int n_nodes, int m_edges, double* d_total_weight,
    struct Edge* d_csr_adj, int *d_csr_node_offset,
    int *community, double *community_degree,
    double *d_vertex_weight,
    int node_to_move, int old_community, int new_community)
{
    double m = *d_total_weight;
    double k_i_in_old = 0.0, k_i_in_new = 0.0;
    // k_i (total degree of node_to_move) is invariant during the kernel — read it
    // in O(1) from the precomputed vertex-weight snapshot instead of summing the
    // adjacency every call.
    double k_i = d_vertex_weight[node_to_move];
    double sum_tot_old = community_degree[old_community];
    double sum_tot_new = community_degree[new_community];

    for (int i = d_csr_node_offset[node_to_move]; i < d_csr_node_offset[node_to_move + 1]; i++) {
        if(d_csr_adj[i].dest == node_to_move)continue;
        if (community[d_csr_adj[i].dest] == old_community)
            k_i_in_old += d_csr_adj[i].weight;
        if (community[d_csr_adj[i].dest] == new_community)
            k_i_in_new += d_csr_adj[i].weight;
    }

    #ifdef DEBUG
    printf("Node %d: k_i_in_old=%f k_i_in_new=%f k_i=%f sum_tot_old=%f sum_tot_new=%f m=%f\n",
           node_to_move, k_i_in_old, k_i_in_new, k_i, sum_tot_old, sum_tot_new, m);
    #endif

    return 2.0 * (k_i_in_new - k_i_in_old) / m
         + 2.0 * ((k_i * (sum_tot_old - sum_tot_new - k_i)) / (m * m));
}


// ============================================================
// KERNELS
// ============================================================

__device__ volatile int changed;
__device__ volatile double delta_Q_sum;
__device__ volatile int louvain_kernel_iteration;

// Louvain local-moving kernel with optional affected vertex support.
// When d_affected is nullptr, all vertices are processed (static behavior).
// When d_affected is provided, only affected vertices are considered,
// and neighbors of moved vertices are marked affected (frontier propagation).
__global__ void louvain_kernel(
    int n_nodes, int m_edges,
    struct Edge* d_csr_adj, int *d_csr_node_offset,
    int *community, double *community_degree,
    double *d_vertex_weight,
    int *d_vertex_locks, int *d_community_locks,
    double* d_total_weight,
    int* d_affected)
{
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int nthreads = blockDim.x * gridDim.x / 32;
    if (threadIdx.x % 32 != 0) return;

    if (tid == 0) changed = 1;
    if (tid == 0) delta_Q_sum = 0.0;
    if (tid == 0) louvain_kernel_iteration = 0;
    grid.sync();

    while (changed && louvain_kernel_iteration < 20) {
        if (tid == 0) changed = 0;
        grid.sync();

        // Node-based iteration with contiguous chunks: each thread processes a
        // local range of nodes, picking the best community among all neighbors.
        long long chunk_start = ((long long)tid * n_nodes) / nthreads;
        long long chunk_end = ((long long)(tid + 1) * n_nodes) / nthreads;
        for (long long node = chunk_start; node < chunk_end; node++) {
            // Dynamic Louvain: skip vertices that are not affected
            if (d_affected != nullptr && d_affected[node] == 0) continue;

            int initial_community = community[node];
            int best_community = initial_community;
            double best_delta_Q = 1e-12;  // threshold

            // Scan all neighbors to find the best target community
            for (int e = d_csr_node_offset[node]; e < d_csr_node_offset[node + 1]; e++) {
                int neighbor = d_csr_adj[e].dest;
                if (neighbor == node) continue;
                int target_comm = community[neighbor];
                if (target_comm == initial_community) continue;

                double dQ = calculate_modularity_change(
                    n_nodes, m_edges, d_total_weight,
                    d_csr_adj, d_csr_node_offset,
                    community, community_degree,
                    d_vertex_weight,
                    node, initial_community, target_comm);

                if (dQ > best_delta_Q) {
                    best_delta_Q = dQ;
                    best_community = target_comm;
                }
            }

            if (best_community != initial_community) {
                // Lock vertex, then re-verify and move
                while (atomicCAS(&d_vertex_locks[node], 0, 1) != 0) {}

                // Re-read community after acquiring lock (may have changed)
                int current_comm = community[node];
                if (current_comm != best_community) {
                    int first_lock = min(current_comm, best_community);
                    int second_lock = max(current_comm, best_community);
                    while (atomicCAS(&d_community_locks[first_lock], 0, 1) != 0) {}
                    while (atomicCAS(&d_community_locks[second_lock], 0, 1) != 0) {}

                    // Re-compute delta_Q under lock for correctness
                    double dQ = calculate_modularity_change(
                        n_nodes, m_edges, d_total_weight,
                        d_csr_adj, d_csr_node_offset,
                        community, community_degree,
                        d_vertex_weight,
                        node, current_comm, best_community);

                    #ifdef DEBUG
                    printf("Node %lld: comm %d -> %d, delta_Q = %f\n",
                           node, current_comm, best_community, dQ);
                    #endif

                    if (dQ > 1e-12) {
                        atomicAdd((double*)&delta_Q_sum, dQ);
                        community[node] = best_community;
                        update_community_degree(n_nodes, d_csr_adj, d_csr_node_offset,
                                                community, community_degree,
                                                node, current_comm, best_community);
                        atomicExch((int*)&changed, 1);

                        // Dynamic: propagate affected flag to neighbors
                        if (d_affected != nullptr) {
                            for (int j = d_csr_node_offset[node];
                                 j < d_csr_node_offset[node + 1]; j++) {
                                d_affected[d_csr_adj[j].dest] = 1;
                            }
                        }
                    }

                    // Ensure community_degree writes are visible to other SMs
                    __threadfence();

                    atomicExch(&d_community_locks[second_lock], 0);
                    atomicExch(&d_community_locks[first_lock], 0);
                }

                atomicExch(&d_vertex_locks[node], 0);
            }
        }

        grid.sync();
        #ifdef DEBUG
        if (tid == 0) {
            printf("Current community assignments:\n");
            for (int i = 0; i < n_nodes; i++)
                printf("  Node %d: Community %d\n", i, community[i]);
        }
        #endif
        if (tid == 0) louvain_kernel_iteration++;
        if (tid == 0) printf("Iteration %d of Louvain kernel completed.\n", louvain_kernel_iteration);
        grid.sync();
    }

    if (tid == 0) printf("Total delta Q in this phase: %f\n", delta_Q_sum);
}


__global__ void count_communities(
    int n_nodes, int *d_community,
    int *d_community_present, int *d_num_communities)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int i = tid; i < n_nodes; i += nthreads)
        atomicExch(&d_community_present[d_community[i]], 1);

    __shared__ int community_count_in_block;
    if (threadIdx.x == 0) community_count_in_block = 0;
    __syncthreads();

    for (int i = tid; i < n_nodes; i += nthreads) {
        if (d_community_present[i] == 1)
            atomicAdd(&community_count_in_block, 1);
    }

    if (threadIdx.x == 0) atomicAdd(d_num_communities, community_count_in_block);
}


__global__ void aggregate_graph(
    int n_nodes, int m_edges,
    struct Edge* d_csr_adj,
    int *d_community_present_prefixsum,
    int *d_community, double *d_community_degree)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    int new_n_nodes = d_community_present_prefixsum[n_nodes - 1];

    for (int i = tid; i < m_edges; i += nthreads) {
        int src = d_csr_adj[i].src;
        int dest = d_csr_adj[i].dest;
        d_csr_adj[i].src = d_community_present_prefixsum[d_community[src]] - 1;
        d_csr_adj[i].dest = d_community_present_prefixsum[d_community[dest]] - 1;
    }
    for (int i = tid; i < new_n_nodes; i += nthreads) {
        d_community_degree[i] = 0.0;
        d_community[i] = i;
    }
    grid.sync();
    for (int i = tid; i < m_edges; i += nthreads) {
        atomicAdd(&d_community_degree[d_csr_adj[i].src], d_csr_adj[i].weight);
    }
}


__global__ void copy_weights_kernel(struct Edge* out_edges, double* out_weights, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += nthreads)
        out_edges[i].weight = out_weights[i];
}


// Track original vertex -> final community mapping through aggregation phases.
// Called BEFORE aggregate_graph (which resets d_community to identity).
// After this kernel: d_original_community[v] = renumbered community of original vertex v.
__global__ void update_original_communities(
    int n_original_nodes,
    int* d_original_community,
    int* d_community,
    int* d_community_present_prefixsum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int i = tid; i < n_original_nodes; i += nthreads) {
        int agg_node = d_original_community[i];
        int comm = d_community[agg_node];
        d_original_community[i] = d_community_present_prefixsum[comm] - 1;
    }
}


// Mark affected vertices using the Frontier approach for dynamic Louvain.
// Deletions: if endpoints were in the same community, mark both as affected
//   (removing an intra-community edge may weaken the community).
// Insertions: if endpoints are in different communities, mark both as affected
//   (adding an inter-community edge may pull vertices to new communities).
__global__ void mark_affected_frontier(
    int n_deletions,
    int* d_del_src, int* d_del_dest,
    int n_insertions,
    int* d_ins_src, int* d_ins_dest,
    int* d_community,
    int* d_affected)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    // Process deletions: mark if both endpoints were in the same community
    for (int i = tid; i < n_deletions; i += nthreads) {
        int u = d_del_src[i];
        int v = d_del_dest[i];
        if (d_community[u] == d_community[v]) {
            d_affected[u] = 1;
            d_affected[v] = 1;
        }
    }

    // Process insertions: mark if endpoints are in different communities
    for (int i = tid; i < n_insertions; i += nthreads) {
        int u = d_ins_src[i];
        int v = d_ins_dest[i];
        if (d_community[u] != d_community[v]) {
            d_affected[u] = 1;
            d_affected[v] = 1;
        }
    }
}


// Compute each vertex's true degree k_i = sum of its incident edge weights,
// read directly from the CSR adjacency. This is the correct per-vertex k_i in
// BOTH regimes the local-moving kernel runs under:
//   - singleton state (static Louvain, and dynamic passes >= 2, where every
//     vertex/super-node is alone in its community): equals community_degree[v];
//   - warm-started dynamic FIRST pass: community_degree holds per-COMMUNITY
//     sums (indexed by community id), so it must NOT be used as the per-vertex
//     k_i — but this recomputation from CSR is still correct.
// O(m) per pass; d_vertex_weight stays constant while the kernel mutates
// community_degree.
__global__ void compute_vertex_weight_kernel(
    int n_nodes, struct Edge* d_csr_adj, int* d_csr_node_offset,
    double* d_vertex_weight)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int v = tid; v < n_nodes; v += nthreads) {
        double s = 0.0;
        for (int e = d_csr_node_offset[v]; e < d_csr_node_offset[v + 1]; e++)
            s += d_csr_adj[e].weight;
        d_vertex_weight[v] = s;
    }
}


// ============================================================
// HOST UTILITY FUNCTIONS
// ============================================================

double calculate_modularity(int n_nodes, int m_edges,
    struct Edge* h_csr_adj, int* h_csr_node_offset, int* community)
{
    double m = 0.0;
    for (int i = 0; i < m_edges; i++) m += h_csr_adj[i].weight;

    double Q = 0.0;
    map<int, double> comm_degree, comm_internal;

    for (int u = 0; u < n_nodes; u++) {
        int comm_u = community[u];
        double degree_u = 0.0;
        for (int i = h_csr_node_offset[u]; i < h_csr_node_offset[u + 1]; i++) {
            degree_u += h_csr_adj[i].weight;
            if (community[h_csr_adj[i].dest] == comm_u)
                comm_internal[comm_u] += h_csr_adj[i].weight;
        }
        comm_degree[comm_u] += degree_u;
    }

    for (const auto& cd_pair : comm_degree) {
        int comm = cd_pair.first;
        double degree = cd_pair.second;
        double internal = comm_internal[comm];
        Q += (internal / m) - (degree / m) * (degree / m);
    }
    return Q;
}


void combine_edges(struct Edge*& d_csr_adj, int& m_edges) {
    thrust::device_ptr<struct Edge> d_csr_adj_ptr = thrust::device_pointer_cast(d_csr_adj);
    thrust::sort(d_csr_adj_ptr, d_csr_adj_ptr + m_edges, EdgeComparator());

    Edge* d_out_edges;
    double* d_out_weights;
    CUDA_CHECK(cudaMalloc(&d_out_edges, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_out_weights, m_edges * sizeof(double)));

    auto key_begin = thrust::make_transform_iterator(d_csr_adj_ptr, GetKey());
    auto weight_begin = thrust::make_transform_iterator(d_csr_adj_ptr, GetWeight());
    auto out_key_begin = thrust::make_transform_output_iterator(
        thrust::device_pointer_cast(d_out_edges), StoreKey());

    auto result = thrust::reduce_by_key(
        key_begin, key_begin + m_edges,
        weight_begin,
        out_key_begin,
        thrust::device_pointer_cast(d_out_weights),
        thrust::equal_to<thrust::pair<int,int>>(),
        thrust::plus<double>()
    );

    int new_m_edges = result.first - out_key_begin;
    cout << "Edges after combining: " << new_m_edges << endl;

    copy_weights_kernel<<<1, 1024>>>(d_out_edges, d_out_weights, new_m_edges);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFree(d_csr_adj));
    d_csr_adj = d_out_edges;
    m_edges = new_m_edges;
    cudaFree(d_out_weights);
}


// Build CSR representation from adjacency list.
// h_csr_adj, h_csr_node_offset, h_community_degree must be pre-allocated.
void build_csr(
    const vector<vector<pair<int, double>>>& adj,
    int n_nodes, int m_edges,
    struct Edge* h_csr_adj, int* h_csr_node_offset,
    double* h_community_degree, double& h_total_weight)
{
    h_total_weight = 0.0;
    for (int i = 0; i < n_nodes; i++) h_community_degree[i] = 0.0;

    int edge_counter = 0;
    for (int i = 0; i < n_nodes; i++) {
        h_csr_node_offset[i] = edge_counter;
        for (const auto& edge : adj[i]) {
            h_csr_adj[edge_counter].src = i;
            h_csr_adj[edge_counter].dest = edge.first;
            h_csr_adj[edge_counter].weight = edge.second;
            edge_counter++;
            h_community_degree[i] += edge.second;
            h_total_weight += edge.second;
        }
    }
    h_csr_node_offset[n_nodes] = m_edges;
}


// Apply batch of edge deletions and insertions to the adjacency list.
// Returns the new number of directed edges.
int apply_batch_updates(
    vector<vector<pair<int, double>>>& adj,
    const vector<tuple<int, int, double>>& deletions,
    const vector<tuple<int, int, double>>& insertions)
{
    // Apply deletions
    for (size_t di = 0; di < deletions.size(); di++) {
        int u = get<0>(deletions[di]);
        int v = get<1>(deletions[di]);
        // Remove edge u -> v
        auto& adj_u = adj[u];
        for (auto it = adj_u.begin(); it != adj_u.end(); ++it) {
            if (it->first == v) { adj_u.erase(it); break; }
        }
        // Remove edge v -> u
        auto& adj_v = adj[v];
        for (auto it = adj_v.begin(); it != adj_v.end(); ++it) {
            if (it->first == u) { adj_v.erase(it); break; }
        }
    }

    // Apply insertions
    for (size_t ii = 0; ii < insertions.size(); ii++) {
        int u = get<0>(insertions[ii]);
        int v = get<1>(insertions[ii]);
        double w = get<2>(insertions[ii]);
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }

    // Count total directed edges
    int m = 0;
    for (const auto& neighbors : adj) m += neighbors.size();
    return m;
}


// Build CSR node offsets from edge array using thrust on device.
void rebuild_csr_offsets(struct Edge* d_csr_adj, int* d_csr_node_offset,
                         int n_nodes, int m_edges)
{
    cudaMemset(d_csr_node_offset, 0, (n_nodes + 1) * sizeof(int));
    thrust::device_ptr<struct Edge> d_csr_adj_ptr = thrust::device_pointer_cast(d_csr_adj);
    thrust::for_each(d_csr_adj_ptr, d_csr_adj_ptr + m_edges,
                     CountEdgesBySrc(d_csr_node_offset));
    thrust::device_ptr<int> dev_ptr_offsets = thrust::device_pointer_cast(d_csr_node_offset);
    thrust::exclusive_scan(dev_ptr_offsets, dev_ptr_offsets + n_nodes + 1, dev_ptr_offsets);
}


// Compute affected vertices using Delta-Screening approach (host-side).
// More selective than Frontier: insertions only mark a vertex if the
// newly inserted edges alone produce a positive modularity gain.
// Three phases:
//   1. Deletions: if endpoints in same community → mark vertex + community
//   2. Insertions: for each source vertex u with inter-community insertions,
//      accumulate weight to each target community, compute best delta modularity;
//      mark vertex + target community only if delta > 0
//   3. Propagation: neighbors of marked vertices + all vertices in marked communities
void compute_affected_delta_screening(
    vector<int>& h_affected,
    int n_nodes,
    const vector<vector<pair<int, double>>>& adj,
    const vector<tuple<int, int, double>>& deletions,
    const vector<tuple<int, int, double>>& insertions,
    const vector<int>& community,
    const vector<double>& vertex_weight,
    const vector<double>& community_weight,
    double total_weight)
{
    double M = total_weight;  // total directed weight (= 2 * undirected weight)

    vector<int> neighbors_flag(n_nodes, 0);
    vector<int> communities_flag(n_nodes, 0);
    h_affected.assign(n_nodes, 0);

    // Phase 1: Deletions — same criterion as Frontier
    // If both endpoints are in the same community, the deletion weakens it.
    for (size_t i = 0; i < deletions.size(); i++) {
        int u = get<0>(deletions[i]);
        int v = get<1>(deletions[i]);
        if (community[u] == community[v]) {
            h_affected[u] = 1;
            h_affected[v] = 1;
            neighbors_flag[u] = 1;
            neighbors_flag[v] = 1;
            communities_flag[community[v]] = 1;
        }
    }

    // Phase 2: Insertions — delta-screening
    // Sort insertions by source vertex so we can process all insertions from
    // the same vertex together and compute per-community weight accumulation.
    vector<tuple<int, int, double>> sorted_ins = insertions;
    sort(sorted_ins.begin(), sorted_ins.end(),
         [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); });

    for (size_t i = 0; i < sorted_ins.size(); ) {
        int u = get<0>(sorted_ins[i]);
        int d = community[u];  // current community of u

        // Accumulate weight from u to each target community via inserted edges
        map<int, double> comm_weight_from_u;  // community c -> weight from u to c
        bool has_inter_community = false;

        for (; i < sorted_ins.size() && get<0>(sorted_ins[i]) == u; i++) {
            int v = get<1>(sorted_ins[i]);
            double w = get<2>(sorted_ins[i]);
            int c = community[v];
            if (c == d) continue;  // skip intra-community insertions
            comm_weight_from_u[c] += w;
            has_inter_community = true;
        }

        if (!has_inter_community) continue;

        // Compute delta modularity for each candidate community
        // deltaModularity = (vcout_c - vcout_d) / M - vtot_u * (vtot_u + ctot_c - ctot_d) / (2*M*M)
        // Here vcout_d = 0 (no inserted edges go to u's own community, we skipped those)
        double k_u = vertex_weight[u];  // total degree of u
        double sigma_d = community_weight[d];  // total weight of u's current community

        int best_comm = -1;
        double best_delta = 0.0;

        for (auto it = comm_weight_from_u.begin(); it != comm_weight_from_u.end(); ++it) {
            int c = it->first;
            double vcout_c = it->second;
            double sigma_c = community_weight[c];

            // Delta modularity: moving u from community d to community c
            // using only the weight contribution of newly inserted edges
            double delta = vcout_c / M - k_u * (k_u + sigma_c - sigma_d) / (2.0 * M * M);

            if (delta > best_delta) {
                best_delta = delta;
                best_comm = c;
            }
        }

        // Only mark u if there exists a community with positive delta
        if (best_delta > 1e-12 && best_comm >= 0) {
            h_affected[u] = 1;
            neighbors_flag[u] = 1;
            communities_flag[best_comm] = 1;
        }
    }

    // Phase 3: Propagation
    // (a) Neighbors of directly-marked vertices are also affected
    for (int u = 0; u < n_nodes; u++) {
        if (neighbors_flag[u]) {
            for (const auto& edge : adj[u]) {
                h_affected[edge.first] = 1;
            }
        }
    }
    // (b) All vertices in an affected community are marked
    for (int u = 0; u < n_nodes; u++) {
        if (communities_flag[community[u]]) {
            h_affected[u] = 1;
        }
    }

    int n_affected = 0;
    for (int u = 0; u < n_nodes; u++) n_affected += h_affected[u];
    cout << "Delta-screening: " << n_affected << " / " << n_nodes
         << " vertices affected" << endl;
}


// ============================================================
// LOUVAIN IMPLEMENTATIONS
// ============================================================

// Run multi-pass Louvain (local-moving + aggregation loop).
// This is the core shared logic for both static and dynamic Louvain.
// Parameters:
//   n_original_nodes: size of the original graph (for d_original_community)
//   d_affected: if not nullptr, used for the FIRST pass only (dynamic)
//               subsequent passes on aggregated graph use all vertices
static void run_louvain_phases(
    int& n_nodes, int& m_edges,
    int n_original_nodes,
    struct Edge* h_csr_adj, int* h_csr_node_offset,
    int* h_community, double* h_community_degree,
    struct Edge*& d_csr_adj, int* d_csr_node_offset,
    int* d_community, double* d_community_degree,
    double* d_vertex_weight,
    int* d_vertex_locks, int* d_community_locks,
    double* d_total_weight,
    int* d_community_present_prefixsum, int* d_num_communities,
    int* d_original_community,
    int* d_affected)
{
    int old_n_nodes = -1;
    bool first_pass = true;
    int pass_count = 0;
    const int MAX_PASSES = 20;

    while (old_n_nodes != n_nodes && pass_count < MAX_PASSES) {
        old_n_nodes = n_nodes;
        pass_count++;

        // Reset locks
        cudaMemset(d_vertex_locks, 0, n_nodes * sizeof(int));
        cudaMemset(d_community_locks, 0, n_nodes * sizeof(int));

        // For the first pass in dynamic mode, use d_affected.
        // For subsequent passes (on aggregated graph), process all vertices.
        int* affected_param = (first_pass && d_affected != nullptr) ? d_affected : nullptr;

        #ifdef DEBUG
        cout << "--- Phase start: n_nodes=" << n_nodes << " m_edges=" << m_edges << " ---" << endl;
        #endif

        // Compute k_i (per-vertex degree) from CSR. Must NOT copy from
        // community_degree: on the warm-started dynamic first pass that array is
        // indexed by community id, not vertex id, which would feed k_i=0 to most
        // vertices and collapse modularity. CSR recomputation is correct in all
        // passes (see compute_vertex_weight_kernel).
        compute_vertex_weight_kernel<<<256, 256>>>(
            n_nodes, d_csr_adj, d_csr_node_offset, d_vertex_weight);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cooperative launch of louvain kernel
        void* kernelArgs[] = {
            (void*)&n_nodes, (void*)&m_edges,
            (void*)&d_csr_adj, (void*)&d_csr_node_offset,
            (void*)&d_community, (void*)&d_community_degree,
            (void*)&d_vertex_weight,
            (void*)&d_vertex_locks, (void*)&d_community_locks,
            (void*)&d_total_weight,
            (void*)&affected_param
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)louvain_kernel, 32, 512, kernelArgs));
        CUDA_CHECK(cudaDeviceSynchronize());
        cout << "Louvain kernel phase completed." << endl;

        // Check if this phase made any meaningful improvement
        double h_delta_Q_sum;
        cudaMemcpyFromSymbol(&h_delta_Q_sum, delta_Q_sum, sizeof(double));
        if (h_delta_Q_sum < 1e-6) {
            cout << "Negligible modularity improvement (" << h_delta_Q_sum << "). Stopping." << endl;
            break;
        }

        first_pass = false;

        // Count communities
        cudaMemset(d_community_present_prefixsum, 0, n_nodes * sizeof(int));
        cudaMemset(d_num_communities, 0, sizeof(int));
        count_communities<<<1, 1024>>>(
            n_nodes, d_community, d_community_present_prefixsum, d_num_communities);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_num_communities;
        cudaMemcpy(&h_num_communities, d_num_communities, sizeof(int), cudaMemcpyDeviceToHost);
        cout << "Number of communities: " << h_num_communities << endl;

        // Prefix sum for renumbering
        thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_community_present_prefixsum);
        thrust::inclusive_scan(dev_ptr, dev_ptr + n_nodes, dev_ptr);

        // Update original community tracking BEFORE aggregation resets d_community
        update_original_communities<<<32, 512>>>(
            n_original_nodes, d_original_community,
            d_community, d_community_present_prefixsum);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Aggregate graph
        void* kernelArgs1[] = {
            (void*)&n_nodes, (void*)&m_edges,
            (void*)&d_csr_adj, (void*)&d_community_present_prefixsum,
            (void*)&d_community, (void*)&d_community_degree
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)aggregate_graph, 32, 512, kernelArgs1));
        CUDA_CHECK(cudaDeviceSynchronize());

        n_nodes = h_num_communities;

        // Copy data from device to host for modularity check
        cudaMemcpy(h_csr_adj, d_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_community, d_community, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_community_degree, d_community_degree, n_nodes * sizeof(double), cudaMemcpyDeviceToHost);

        #ifdef DEBUG
        for (int i = 0; i < n_nodes; i++)
            cout << "  Partition " << i << " -> community " << h_community[i] << endl;
        for (int i = 0; i < m_edges; i++)
            cout << "  Edge " << i << ": " << h_csr_adj[i].src << " -> "
                 << h_csr_adj[i].dest << " w=" << h_csr_adj[i].weight << endl;
        #endif

        // Combine edges with same src and dest (aggregation creates duplicates)
        combine_edges(d_csr_adj, m_edges);

        cudaMemcpy(h_csr_adj, d_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_community, d_community, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_community_degree, d_community_degree, n_nodes * sizeof(double), cudaMemcpyDeviceToHost);

        // Rebuild CSR node offsets for the aggregated graph
        rebuild_csr_offsets(d_csr_adj, d_csr_node_offset, n_nodes, m_edges);

        cudaMemcpy(h_csr_node_offset, d_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community)
             << " is the current modularity." << endl;
    }
}


// Run static Louvain on the initial graph and return a LouvainState
// that can be used for subsequent dynamic runs.
LouvainState louvain_static_cuda(
    const vector<vector<pair<int, double>>>& adj,
    int n_nodes, int m_edges)
{
    int n_original_nodes = n_nodes;

    // Allocate host arrays
    struct Edge* h_csr_adj = new struct Edge[m_edges];
    int* h_csr_node_offset = new int[n_nodes + 1];
    int* h_community = new int[n_nodes];
    double* h_community_degree = new double[n_nodes];
    double h_total_weight = 0.0;

    // Build CSR
    build_csr(adj, n_nodes, m_edges, h_csr_adj, h_csr_node_offset,
              h_community_degree, h_total_weight);
    // Initialize each vertex in its own community
    for (int i = 0; i < n_nodes; i++) h_community[i] = i;

    // Allocate device memory
    struct Edge* d_csr_adj;
    int* d_csr_node_offset;
    int* d_community;
    int* d_community_locks;
    int* d_vertex_locks;
    double* d_total_weight;
    double* d_community_degree;
    int* d_community_present_prefixsum;
    int* d_num_communities;
    int* d_original_community;

    CUDA_CHECK(cudaMalloc(&d_csr_adj, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_csr_node_offset, (n_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_degree, n_nodes * sizeof(double)));
    double* d_vertex_weight;
    CUDA_CHECK(cudaMalloc(&d_vertex_weight, n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vertex_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_weight, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_community_present_prefixsum, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_communities, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_original_community, n_nodes * sizeof(int)));

    // Upload data to device
    cudaMemcpy(d_csr_adj, h_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_node_offset, h_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, h_community, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_degree, h_community_degree, n_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_weight, &h_total_weight, sizeof(double), cudaMemcpyHostToDevice);
    // Initialize original community mapping to identity
    thrust::device_ptr<int> d_orig_comm_ptr = thrust::device_pointer_cast(d_original_community);
    thrust::sequence(d_orig_comm_ptr, d_orig_comm_ptr + n_nodes);

    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community)
         << " is the initial modularity." << endl;

    // Run multi-pass Louvain (static: d_affected = nullptr)
    auto start = chrono::steady_clock::now();

    run_louvain_phases(
        n_nodes, m_edges, n_original_nodes,
        h_csr_adj, h_csr_node_offset, h_community, h_community_degree,
        d_csr_adj, d_csr_node_offset, d_community, d_community_degree,
        d_vertex_weight,
        d_vertex_locks, d_community_locks, d_total_weight,
        d_community_present_prefixsum, d_num_communities,
        d_original_community,
        nullptr  // static: no affected flag
    );

    auto end = chrono::steady_clock::now();

    // Build result state
    LouvainState state;
    state.n_nodes = n_original_nodes;
    state.community.resize(n_original_nodes);
    state.vertex_weight.resize(n_original_nodes, 0.0);
    state.community_weight.resize(n_original_nodes, 0.0);

    // Download original community assignments
    cudaMemcpy(state.community.data(), d_original_community,
               n_original_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute vertex_weight and community_weight from the ORIGINAL graph CSR
    // (rebuild CSR since h_csr_adj/h_csr_node_offset now hold the aggregated graph)
    {
        int orig_m = 0;
        for (const auto& neighbors : adj) orig_m += neighbors.size();
        struct Edge* orig_csr = new struct Edge[orig_m];
        int* orig_offsets = new int[n_original_nodes + 1];
        double* dummy_cd = new double[n_original_nodes];
        double dummy_tw;
        build_csr(adj, n_original_nodes, orig_m, orig_csr, orig_offsets, dummy_cd, dummy_tw);
        state.total_weight = dummy_tw;

        for (int v = 0; v < n_original_nodes; v++) {
            state.vertex_weight[v] = dummy_cd[v]; // dummy_cd[v] = sum of edge weights for vertex v
            state.community_weight[state.community[v]] += dummy_cd[v];
        }

        delete[] orig_csr;
        delete[] orig_offsets;
        delete[] dummy_cd;
    }

    cout << "Static Louvain: final " << n_nodes << " communities, "
         << m_edges << " edges" << endl;
    cout << "Static execution time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Print community assignment summary
    {
        map<int, int> comm_sizes;
        for (int v = 0; v < n_original_nodes; v++)
            comm_sizes[state.community[v]]++;
        cout << "Community sizes: ";
        for (auto it = comm_sizes.begin(); it != comm_sizes.end(); ++it)
            cout << it->first << ":" << it->second << " ";
        cout << endl;
    }

    // Free device memory
    cudaFree(d_csr_adj);
    cudaFree(d_csr_node_offset);
    cudaFree(d_community);
    cudaFree(d_community_degree);
    cudaFree(d_vertex_weight);
    cudaFree(d_vertex_locks);
    cudaFree(d_community_locks);
    cudaFree(d_total_weight);
    cudaFree(d_community_present_prefixsum);
    cudaFree(d_num_communities);
    cudaFree(d_original_community);

    // Free host memory
    delete[] h_csr_adj;
    delete[] h_csr_node_offset;
    delete[] h_community;
    delete[] h_community_degree;

    return state;
}


// Run Naive Dynamic Louvain: start from previous community assignments,
// but process ALL vertices (no affected vertex optimization).
// This is faster than static because the starting point is already good.
LouvainState louvain_dynamic_naive_cuda(
    const vector<vector<pair<int, double>>>& adj,
    int n_nodes, int m_edges,
    const LouvainState& prev_state)
{
    int n_original_nodes = n_nodes;
    assert(n_nodes == prev_state.n_nodes);

    // Allocate host arrays
    struct Edge* h_csr_adj = new struct Edge[m_edges];
    int* h_csr_node_offset = new int[n_nodes + 1];
    int* h_community = new int[n_nodes];
    double* h_community_degree = new double[n_nodes];
    double h_total_weight = 0.0;

    // Build CSR from updated adjacency list
    build_csr(adj, n_nodes, m_edges, h_csr_adj, h_csr_node_offset,
              h_community_degree, h_total_weight);

    // Initialize community from previous state
    for (int i = 0; i < n_nodes; i++) h_community[i] = prev_state.community[i];

    // Recompute community_degree based on new graph + old community assignments
    // community_degree[c] = sum of vertex_weight[v] for all v in community c
    memset(h_community_degree, 0, n_nodes * sizeof(double));
    for (int v = 0; v < n_nodes; v++) {
        double vw = 0.0;
        for (int j = h_csr_node_offset[v]; j < h_csr_node_offset[v + 1]; j++)
            vw += h_csr_adj[j].weight;
        h_community_degree[h_community[v]] += vw;
    }

    // Allocate device memory
    struct Edge* d_csr_adj;
    int* d_csr_node_offset;
    int* d_community;
    int* d_community_locks;
    int* d_vertex_locks;
    double* d_total_weight;
    double* d_community_degree;
    int* d_community_present_prefixsum;
    int* d_num_communities;
    int* d_original_community;

    CUDA_CHECK(cudaMalloc(&d_csr_adj, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_csr_node_offset, (n_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_degree, n_nodes * sizeof(double)));
    double* d_vertex_weight;
    CUDA_CHECK(cudaMalloc(&d_vertex_weight, n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vertex_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_weight, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_community_present_prefixsum, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_communities, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_original_community, n_nodes * sizeof(int)));

    // Upload data to device
    cudaMemcpy(d_csr_adj, h_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_node_offset, h_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, h_community, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_degree, h_community_degree, n_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_weight, &h_total_weight, sizeof(double), cudaMemcpyHostToDevice);
    // Initialize original community mapping to identity
    thrust::device_ptr<int> d_orig_comm_ptr = thrust::device_pointer_cast(d_original_community);
    thrust::sequence(d_orig_comm_ptr, d_orig_comm_ptr + n_nodes);

    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community)
         << " is the modularity before naive-dynamic Louvain." << endl;

    auto start = chrono::steady_clock::now();

    // Run multi-pass Louvain (naive dynamic: d_affected = nullptr, processes all vertices)
    run_louvain_phases(
        n_nodes, m_edges, n_original_nodes,
        h_csr_adj, h_csr_node_offset, h_community, h_community_degree,
        d_csr_adj, d_csr_node_offset, d_community, d_community_degree,
        d_vertex_weight,
        d_vertex_locks, d_community_locks, d_total_weight,
        d_community_present_prefixsum, d_num_communities,
        d_original_community,
        nullptr  // naive: no affected flag, process all vertices
    );

    auto end = chrono::steady_clock::now();

    // Build result state
    LouvainState state;
    state.n_nodes = n_original_nodes;
    state.community.resize(n_original_nodes);
    state.vertex_weight.resize(n_original_nodes, 0.0);
    state.community_weight.resize(n_original_nodes, 0.0);

    cudaMemcpy(state.community.data(), d_original_community,
               n_original_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Recompute vertex and community weights from current graph
    {
        int orig_m = 0;
        for (const auto& neighbors : adj) orig_m += neighbors.size();
        struct Edge* orig_csr = new struct Edge[orig_m];
        int* orig_offsets = new int[n_original_nodes + 1];
        double* vw = new double[n_original_nodes];
        double tw;
        build_csr(adj, n_original_nodes, orig_m, orig_csr, orig_offsets, vw, tw);
        state.total_weight = tw;
        for (int v = 0; v < n_original_nodes; v++) {
            state.vertex_weight[v] = vw[v];
            state.community_weight[state.community[v]] += vw[v];
        }
        delete[] orig_csr;
        delete[] orig_offsets;
        delete[] vw;
    }

    cout << "Naive-Dynamic Louvain: final " << n_nodes << " communities" << endl;
    cout << "Naive-Dynamic execution time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Free device memory
    cudaFree(d_csr_adj);
    cudaFree(d_csr_node_offset);
    cudaFree(d_community);
    cudaFree(d_community_degree);
    cudaFree(d_vertex_weight);
    cudaFree(d_vertex_locks);
    cudaFree(d_community_locks);
    cudaFree(d_total_weight);
    cudaFree(d_community_present_prefixsum);
    cudaFree(d_num_communities);
    cudaFree(d_original_community);

    delete[] h_csr_adj;
    delete[] h_csr_node_offset;
    delete[] h_community;
    delete[] h_community_degree;

    return state;
}


// Run Frontier-based Dynamic Louvain: start from previous community assignments,
// but only process vertices affected by the batch update.
// Affected vertices are those at the boundary of changed edges:
//   - Deletions where both endpoints were in the same community
//   - Insertions where endpoints are in different communities
// During local-moving, neighbors of moved vertices also become affected.
LouvainState louvain_dynamic_frontier_cuda(
    const vector<vector<pair<int, double>>>& adj,
    int n_nodes, int m_edges,
    const LouvainState& prev_state,
    const vector<tuple<int, int, double>>& deletions,
    const vector<tuple<int, int, double>>& insertions)
{
    int n_original_nodes = n_nodes;
    int n_deletions = deletions.size();
    int n_insertions = insertions.size();
    assert(n_nodes == prev_state.n_nodes);

    // Allocate host arrays
    struct Edge* h_csr_adj = new struct Edge[m_edges];
    int* h_csr_node_offset = new int[n_nodes + 1];
    int* h_community = new int[n_nodes];
    double* h_community_degree = new double[n_nodes];
    double h_total_weight = 0.0;

    // Build CSR from updated adjacency list
    build_csr(adj, n_nodes, m_edges, h_csr_adj, h_csr_node_offset,
              h_community_degree, h_total_weight);

    // Initialize community from previous state
    for (int i = 0; i < n_nodes; i++) h_community[i] = prev_state.community[i];

    // Recompute community_degree based on new graph + old community assignments
    memset(h_community_degree, 0, n_nodes * sizeof(double));
    for (int v = 0; v < n_nodes; v++) {
        double vw = 0.0;
        for (int j = h_csr_node_offset[v]; j < h_csr_node_offset[v + 1]; j++)
            vw += h_csr_adj[j].weight;
        h_community_degree[h_community[v]] += vw;
    }

    // Prepare batch edge arrays for GPU marking
    vector<int> h_del_src(n_deletions), h_del_dest(n_deletions);
    vector<int> h_ins_src(n_insertions), h_ins_dest(n_insertions);
    for (int i = 0; i < n_deletions; i++) {
        h_del_src[i] = get<0>(deletions[i]);
        h_del_dest[i] = get<1>(deletions[i]);
    }
    for (int i = 0; i < n_insertions; i++) {
        h_ins_src[i] = get<0>(insertions[i]);
        h_ins_dest[i] = get<1>(insertions[i]);
    }

    // Allocate device memory
    struct Edge* d_csr_adj;
    int* d_csr_node_offset;
    int* d_community;
    int* d_community_locks;
    int* d_vertex_locks;
    double* d_total_weight;
    double* d_community_degree;
    int* d_community_present_prefixsum;
    int* d_num_communities;
    int* d_original_community;
    int* d_affected;

    CUDA_CHECK(cudaMalloc(&d_csr_adj, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_csr_node_offset, (n_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_degree, n_nodes * sizeof(double)));
    double* d_vertex_weight;
    CUDA_CHECK(cudaMalloc(&d_vertex_weight, n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vertex_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_weight, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_community_present_prefixsum, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_communities, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_original_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_affected, n_nodes * sizeof(int)));
    cudaMemset(d_affected, 0, n_nodes * sizeof(int));

    // Allocate and upload batch edge arrays
    int* d_del_src = nullptr;
    int* d_del_dest = nullptr;
    int* d_ins_src = nullptr;
    int* d_ins_dest = nullptr;

    if (n_deletions > 0) {
        CUDA_CHECK(cudaMalloc(&d_del_src, n_deletions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_del_dest, n_deletions * sizeof(int)));
        cudaMemcpy(d_del_src, h_del_src.data(), n_deletions * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_del_dest, h_del_dest.data(), n_deletions * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (n_insertions > 0) {
        CUDA_CHECK(cudaMalloc(&d_ins_src, n_insertions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ins_dest, n_insertions * sizeof(int)));
        cudaMemcpy(d_ins_src, h_ins_src.data(), n_insertions * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ins_dest, h_ins_dest.data(), n_insertions * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Upload graph data to device
    cudaMemcpy(d_csr_adj, h_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_node_offset, h_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, h_community, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_degree, h_community_degree, n_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_weight, &h_total_weight, sizeof(double), cudaMemcpyHostToDevice);
    // Initialize original community mapping to identity
    thrust::device_ptr<int> d_orig_comm_ptr = thrust::device_pointer_cast(d_original_community);
    thrust::sequence(d_orig_comm_ptr, d_orig_comm_ptr + n_nodes);

    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community)
         << " is the modularity before frontier-dynamic Louvain." << endl;

    auto start = chrono::steady_clock::now();

    // Mark affected vertices using frontier approach
    if (n_deletions > 0 || n_insertions > 0) {
        mark_affected_frontier<<<32, 256>>>(
            n_deletions, d_del_src, d_del_dest,
            n_insertions, d_ins_src, d_ins_dest,
            d_community, d_affected);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Count affected vertices for reporting
    {
        thrust::device_ptr<int> d_aff_ptr = thrust::device_pointer_cast(d_affected);
        int n_affected = thrust::reduce(d_aff_ptr, d_aff_ptr + n_nodes);
        cout << "Frontier marking: " << n_affected << " / " << n_nodes
             << " vertices affected" << endl;
    }

    // Run multi-pass Louvain with affected vertex optimization
    run_louvain_phases(
        n_nodes, m_edges, n_original_nodes,
        h_csr_adj, h_csr_node_offset, h_community, h_community_degree,
        d_csr_adj, d_csr_node_offset, d_community, d_community_degree,
        d_vertex_weight,
        d_vertex_locks, d_community_locks, d_total_weight,
        d_community_present_prefixsum, d_num_communities,
        d_original_community,
        d_affected  // frontier: only process affected vertices in first pass
    );

    auto end = chrono::steady_clock::now();

    // Build result state
    LouvainState state;
    state.n_nodes = n_original_nodes;
    state.community.resize(n_original_nodes);
    state.vertex_weight.resize(n_original_nodes, 0.0);
    state.community_weight.resize(n_original_nodes, 0.0);

    cudaMemcpy(state.community.data(), d_original_community,
               n_original_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Recompute weights from original graph
    {
        int orig_m = 0;
        for (const auto& neighbors : adj) orig_m += neighbors.size();
        struct Edge* orig_csr = new struct Edge[orig_m];
        int* orig_offsets = new int[n_original_nodes + 1];
        double* vw = new double[n_original_nodes];
        double tw;
        build_csr(adj, n_original_nodes, orig_m, orig_csr, orig_offsets, vw, tw);
        state.total_weight = tw;
        for (int v = 0; v < n_original_nodes; v++) {
            state.vertex_weight[v] = vw[v];
            state.community_weight[state.community[v]] += vw[v];
        }
        delete[] orig_csr;
        delete[] orig_offsets;
        delete[] vw;
    }

    cout << "Frontier-Dynamic Louvain: final " << n_nodes << " communities" << endl;
    cout << "Frontier-Dynamic execution time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Print community assignment summary
    {
        map<int, int> comm_sizes;
        for (int v = 0; v < n_original_nodes; v++)
            comm_sizes[state.community[v]]++;
        cout << "Community sizes: ";
        for (auto it = comm_sizes.begin(); it != comm_sizes.end(); ++it)
            cout << it->first << ":" << it->second << " ";
        cout << endl;
    }

    // Free device memory
    cudaFree(d_csr_adj);
    cudaFree(d_csr_node_offset);
    cudaFree(d_community);
    cudaFree(d_community_degree);
    cudaFree(d_vertex_weight);
    cudaFree(d_vertex_locks);
    cudaFree(d_community_locks);
    cudaFree(d_total_weight);
    cudaFree(d_community_present_prefixsum);
    cudaFree(d_num_communities);
    cudaFree(d_original_community);
    cudaFree(d_affected);
    if (d_del_src) cudaFree(d_del_src);
    if (d_del_dest) cudaFree(d_del_dest);
    if (d_ins_src) cudaFree(d_ins_src);
    if (d_ins_dest) cudaFree(d_ins_dest);

    delete[] h_csr_adj;
    delete[] h_csr_node_offset;
    delete[] h_community;
    delete[] h_community_degree;

    return state;
}


// Run Delta-Screening Dynamic Louvain: start from previous community assignments,
// but only process vertices identified by the delta-screening heuristic.
// Delta-screening is more selective than frontier:
//   - Deletions: same as frontier (intra-community edge removal)
//   - Insertions: vertex is only marked affected if the newly inserted edges
//     alone produce a positive modularity gain for moving to another community.
//   - Propagation: neighbors of marked vertices + entire affected communities.
// This typically results in fewer affected vertices than frontier, leading to
// faster convergence at the cost of slightly more complex screening logic.
LouvainState louvain_dynamic_delta_screening_cuda(
    const vector<vector<pair<int, double>>>& adj,
    int n_nodes, int m_edges,
    const LouvainState& prev_state,
    const vector<tuple<int, int, double>>& deletions,
    const vector<tuple<int, int, double>>& insertions)
{
    int n_original_nodes = n_nodes;
    assert(n_nodes == prev_state.n_nodes);

    // Allocate host arrays
    struct Edge* h_csr_adj = new struct Edge[m_edges];
    int* h_csr_node_offset = new int[n_nodes + 1];
    int* h_community = new int[n_nodes];
    double* h_community_degree = new double[n_nodes];
    double h_total_weight = 0.0;

    // Build CSR from updated adjacency list
    build_csr(adj, n_nodes, m_edges, h_csr_adj, h_csr_node_offset,
              h_community_degree, h_total_weight);

    // Initialize community from previous state
    for (int i = 0; i < n_nodes; i++) h_community[i] = prev_state.community[i];

    // Recompute community_degree based on new graph + old community assignments
    memset(h_community_degree, 0, n_nodes * sizeof(double));
    for (int v = 0; v < n_nodes; v++) {
        double vw = 0.0;
        for (int j = h_csr_node_offset[v]; j < h_csr_node_offset[v + 1]; j++)
            vw += h_csr_adj[j].weight;
        h_community_degree[h_community[v]] += vw;
    }

    // Compute affected vertices using delta-screening on the host
    vector<int> h_affected;
    compute_affected_delta_screening(
        h_affected, n_nodes, adj,
        deletions, insertions,
        prev_state.community,
        prev_state.vertex_weight,
        prev_state.community_weight,
        prev_state.total_weight);

    // Allocate device memory
    struct Edge* d_csr_adj;
    int* d_csr_node_offset;
    int* d_community;
    int* d_community_locks;
    int* d_vertex_locks;
    double* d_total_weight;
    double* d_community_degree;
    int* d_community_present_prefixsum;
    int* d_num_communities;
    int* d_original_community;
    int* d_affected;

    CUDA_CHECK(cudaMalloc(&d_csr_adj, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_csr_node_offset, (n_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_degree, n_nodes * sizeof(double)));
    double* d_vertex_weight;
    CUDA_CHECK(cudaMalloc(&d_vertex_weight, n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vertex_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_community_locks, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_weight, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_community_present_prefixsum, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_communities, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_original_community, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_affected, n_nodes * sizeof(int)));

    // Upload graph data to device
    cudaMemcpy(d_csr_adj, h_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_node_offset, h_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, h_community, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_degree, h_community_degree, n_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_weight, &h_total_weight, sizeof(double), cudaMemcpyHostToDevice);
    // Upload precomputed affected array
    cudaMemcpy(d_affected, h_affected.data(), n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    // Initialize original community mapping to identity
    thrust::device_ptr<int> d_orig_comm_ptr = thrust::device_pointer_cast(d_original_community);
    thrust::sequence(d_orig_comm_ptr, d_orig_comm_ptr + n_nodes);

    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community)
         << " is the modularity before delta-screening-dynamic Louvain." << endl;

    auto start = chrono::steady_clock::now();

    // Run multi-pass Louvain with delta-screening affected vertex optimization
    run_louvain_phases(
        n_nodes, m_edges, n_original_nodes,
        h_csr_adj, h_csr_node_offset, h_community, h_community_degree,
        d_csr_adj, d_csr_node_offset, d_community, d_community_degree,
        d_vertex_weight,
        d_vertex_locks, d_community_locks, d_total_weight,
        d_community_present_prefixsum, d_num_communities,
        d_original_community,
        d_affected  // delta-screening: only process screened vertices in first pass
    );

    auto end = chrono::steady_clock::now();

    // Build result state
    LouvainState state;
    state.n_nodes = n_original_nodes;
    state.community.resize(n_original_nodes);
    state.vertex_weight.resize(n_original_nodes, 0.0);
    state.community_weight.resize(n_original_nodes, 0.0);

    cudaMemcpy(state.community.data(), d_original_community,
               n_original_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Recompute weights from original graph
    {
        int orig_m = 0;
        for (const auto& neighbors : adj) orig_m += neighbors.size();
        struct Edge* orig_csr = new struct Edge[orig_m];
        int* orig_offsets = new int[n_original_nodes + 1];
        double* vw = new double[n_original_nodes];
        double tw;
        build_csr(adj, n_original_nodes, orig_m, orig_csr, orig_offsets, vw, tw);
        state.total_weight = tw;
        for (int v = 0; v < n_original_nodes; v++) {
            state.vertex_weight[v] = vw[v];
            state.community_weight[state.community[v]] += vw[v];
        }
        delete[] orig_csr;
        delete[] orig_offsets;
        delete[] vw;
    }

    cout << "Delta-Screening-Dynamic Louvain: final " << n_nodes << " communities" << endl;
    cout << "Delta-Screening-Dynamic execution time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Print community assignment summary
    {
        map<int, int> comm_sizes;
        for (int v = 0; v < n_original_nodes; v++)
            comm_sizes[state.community[v]]++;
        cout << "Community sizes: ";
        for (auto it = comm_sizes.begin(); it != comm_sizes.end(); ++it)
            cout << it->first << ":" << it->second << " ";
        cout << endl;
    }

    // Free device memory
    cudaFree(d_csr_adj);
    cudaFree(d_csr_node_offset);
    cudaFree(d_community);
    cudaFree(d_community_degree);
    cudaFree(d_vertex_weight);
    cudaFree(d_vertex_locks);
    cudaFree(d_community_locks);
    cudaFree(d_total_weight);
    cudaFree(d_community_present_prefixsum);
    cudaFree(d_num_communities);
    cudaFree(d_original_community);
    cudaFree(d_affected);

    delete[] h_csr_adj;
    delete[] h_csr_node_offset;
    delete[] h_community;
    delete[] h_community_degree;

    return state;
}


void write_communities(const char* filename, const LouvainState& state) {
    set<int> unique_comms(state.community.begin(), state.community.end());
    ofstream fout(filename);
    fout << unique_comms.size() << "\n";
    for (int i = 0; i < state.n_nodes; i++)
        fout << i << " " << state.community[i] << "\n";
    fout.close();
    cout << "Community assignments written to " << filename << endl;
}

// ============================================================
// MAIN: Demonstrates static Louvain followed by dynamic updates
// ============================================================

int main(int argc, char* argv[]) {
    // ---- Phase 1: Read initial graph ----
    int n_nodes, m_edges;
    cin >> n_nodes >> m_edges;
    vector<vector<pair<int, double>>> adj(n_nodes);

    for (int i = 0; i < m_edges; i++) {
        int u, v;
        double w;
        cin >> u >> v; // >> w;
        w = 1.0;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }

    int m_directed = 0;
    for (const auto& neighbors : adj) m_directed += neighbors.size();

    cout << "===== STATIC LOUVAIN =====" << endl;
    cout << "Graph: " << n_nodes << " nodes, " << m_directed << " directed edges" << endl;

    LouvainState state = louvain_static_cuda(adj, n_nodes, m_directed);

    // Write static community assignments
    const char* output_file = (argc > 1) ? argv[1] : nullptr;
    if (output_file) write_communities(output_file, state);

    // ---- Phase 2: Dynamic updates ----
    int n_batches;
    if (!(cin >> n_batches)) {
        cout << "No batch updates provided. Done." << endl;
        return 0;
    }

    for (int b = 0; b < n_batches; b++) {
        cout << "\n===== BATCH " << (b + 1) << " / " << n_batches << " =====" << endl;

        int n_del, n_ins;
        cin >> n_del >> n_ins;

        vector<tuple<int, int, double>> deletions(n_del);
        vector<tuple<int, int, double>> insertions(n_ins);

        for (int i = 0; i < n_del; i++) {
            int u, v;
            double w;
            cin >> u >> v; // >> w;
            w = 1;
            deletions[i] = {u, v, w};
        }
        for (int i = 0; i < n_ins; i++) {
            int u, v;
            double w;
            cin >> u >> v; // >> w;
            w = 1;
            insertions[i] = {u, v, w};
        }

        cout << "Batch: " << n_del << " deletions, " << n_ins << " insertions" << endl;

        // Apply batch updates to adjacency list
        m_directed = apply_batch_updates(adj, deletions, insertions);
        cout << "Updated graph: " << n_nodes << " nodes, " << m_directed << " directed edges" << endl;

        // Calculate modularity on the updated graph with OLD community assignments
        {
            struct Edge* tmp_csr = new struct Edge[m_directed];
            int* tmp_offsets = new int[n_nodes + 1];
            double* tmp_cd = new double[n_nodes];
            double tmp_tw;
            build_csr(adj, n_nodes, m_directed, tmp_csr, tmp_offsets, tmp_cd, tmp_tw);
            cout << "Modularity with old communities on new graph: "
                 << calculate_modularity(n_nodes, m_directed, tmp_csr, tmp_offsets,
                                         state.community.data())
                 << endl;
            delete[] tmp_csr;
            delete[] tmp_offsets;
            delete[] tmp_cd;
        }

        // Run all four approaches for comparison
        cout << "\n--- Naive Dynamic Louvain ---" << endl;
        LouvainState state_naive = louvain_dynamic_naive_cuda(
            adj, n_nodes, m_directed, state);

        cout << "\n--- Frontier Dynamic Louvain ---" << endl;
        LouvainState state_frontier = louvain_dynamic_frontier_cuda(
            adj, n_nodes, m_directed, state, deletions, insertions);

        cout << "\n--- Delta-Screening Dynamic Louvain ---" << endl;
        LouvainState state_delta = louvain_dynamic_delta_screening_cuda(
            adj, n_nodes, m_directed, state, deletions, insertions);

        // Compute final modularities for comparison
        {
            struct Edge* tmp_csr = new struct Edge[m_directed];
            int* tmp_offsets = new int[n_nodes + 1];
            double* tmp_cd = new double[n_nodes];
            double tmp_tw;
            build_csr(adj, n_nodes, m_directed, tmp_csr, tmp_offsets, tmp_cd, tmp_tw);

            cout << "\n--- Final Comparison ---" << endl;
            cout << "Naive modularity:           "
                 << calculate_modularity(n_nodes, m_directed, tmp_csr, tmp_offsets,
                                         state_naive.community.data())
                 << endl;
            cout << "Frontier modularity:        "
                 << calculate_modularity(n_nodes, m_directed, tmp_csr, tmp_offsets,
                                         state_frontier.community.data())
                 << endl;
            cout << "Delta-Screening modularity: "
                 << calculate_modularity(n_nodes, m_directed, tmp_csr, tmp_offsets,
                                         state_delta.community.data())
                 << endl;

            delete[] tmp_csr;
            delete[] tmp_offsets;
            delete[] tmp_cd;
        }

        // Use delta-screening result as the starting point for the next batch
        state = state_delta;
    }

    cout << "\nDone." << endl;
    return 0;
}
