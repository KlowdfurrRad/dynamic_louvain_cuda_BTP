// Code for cuda static louvain
#include <vector>
using namespace std;

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <chrono>
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
#include <thrust/iterator/transform_output_iterator.h> // For thrust iterators

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

struct Edge {
    int src;
    int dest;
    double weight;
};

struct EdgeComparator {
    __host__ __device__
    bool operator()(const struct Edge &a, const struct Edge &b) const {
        if (a.src == b.src) {
            return a.dest < b.dest;
        }
        return a.src < b.src;
    }
};

struct GetKey
{
    __host__ __device__
    thrust::pair<int,int> operator()(const Edge &e) const {
        return {e.src, e.dest};
    }
};

struct GetWeight
{
    __host__ __device__
    double operator()(const Edge &e) const {
        return e.weight;
    }
};

struct StoreKey
{
    __host__ __device__
    Edge operator()(const thrust::pair<int,int> &p) const {
        return Edge{p.first, p.second, 0.0}; // weight filled later
    }
};

struct CountEdgesBySrc
{
    int* offsets;   // device pointer to CSR offset array

    // Constructor (host-side)
    CountEdgesBySrc(int* o) : offsets(o) {}

    __device__ void operator()(const Edge& e) const
    {
        // We increment offsets[src + 1] because of CSR convention
        atomicAdd(&offsets[e.src], 1);
    }
};

__device__ double update_community_degree(int n_nodes, struct Edge *d_csr_adj, int *d_csr_node_offset, int *community, double *community_degree, int node_to_move, int old_community, int new_community) {
    for(int i = d_csr_node_offset[node_to_move]; i < d_csr_node_offset[node_to_move + 1]; i++){
        community_degree[old_community] -= d_csr_adj[i].weight;
        community_degree[new_community] += d_csr_adj[i].weight;
        // if(community[d_csr_adj[i].dest] == old_community){
        //     community_degree[old_community] -= d_csr_adj[i].weight;
        // }
        // if(community[d_csr_adj[i].dest] == new_community){
        //     community_degree[new_community] += d_csr_adj[i].weight;
        // }
    }
    return 0.0;
}

__device__ double calculate_modularity_change(
        int n_nodes, 
        int m_edges, 
        double* d_total_weight, 
        struct Edge* d_csr_adj, 
        int *d_csr_node_offset, 
        int *community, 
        double *community_degree, 
        int node_to_move, 
        int old_community, 
        int new_community) 
{
    double m = *d_total_weight;

    double delta_Q = 0.0;
    double k_i_in_old = 0.0, k_i_in_new = 0.0;
    double k_i = 0.0;
    double sum_tot_old = community_degree[old_community], sum_tot_new = community_degree[new_community];

    for(int i = d_csr_node_offset[node_to_move]; i < d_csr_node_offset[node_to_move + 1]; i++){
        k_i += d_csr_adj[i].weight;
        if(community[d_csr_adj[i].dest] == old_community){
            k_i_in_old += d_csr_adj[i].weight;       
        }
        if(community[d_csr_adj[i].dest] == new_community){
            k_i_in_new += d_csr_adj[i].weight;    
        }
    }


    // change to printf for cuda debugging
    #ifdef DEBUG
    printf("Node %d: k_i_in_old = %f, k_i_in_new = %f, k_i = %f, sum_tot_old = %f, sum_tot_new = %f, m = %f\n", node_to_move, k_i_in_old, k_i_in_new, k_i, sum_tot_old, sum_tot_new, m);
    printf("%f This is internal changes\n", 2.0 * (k_i_in_new - k_i_in_old) / m);
    printf("sum_tot_old: %f sum_tot_new: %f k_i: %f\n", sum_tot_old, sum_tot_new, k_i);
    printf("%f This is degree changes\n", 2.0 * ((k_i * (sum_tot_old - sum_tot_new - k_i)) / (m * m)));
    #endif

    delta_Q = 2.0 * (k_i_in_new - k_i_in_old) / m + 2.0 * ((k_i * (sum_tot_old - sum_tot_new - k_i)) / (m * m));
    return delta_Q;
}

__device__ volatile int changed;
__device__ volatile double delta_Q_sum;
__device__ volatile int louvain_kernel_iteration;

// kernel created for launch on only one block
// To allow multiple blocks, we need to use global memory to store changed flag
__global__ void louvain_kernel(
    int n_nodes,
    int m_edges,
    struct Edge* d_csr_adj,
    int *d_csr_node_offset,
    int *community,
    double *community_degree,
    int *d_vertex_locks,
    int *d_community_locks,
    double* d_total_weight ) 
{
    cg::grid_group grid = cg::this_grid();

    // Added /32 because only want one thread working per warp to avoid deadlock type situation
    int tid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int nthreads = blockDim.x * gridDim.x / 32;
    if(threadIdx.x % 32 != 0) return;

    if(tid == 0) changed = 1;
    if(tid == 0) delta_Q_sum = 0.0; //REMOVE ONLY TEMPORARY
    if(tid == 0) louvain_kernel_iteration = 0;
    grid.sync();

    while(changed && louvain_kernel_iteration < 100) {
        if(tid == 0) changed = 0;
        grid.sync();

        // Separate into sections and move
        // We discussed that vertex parallelism may be useful but came to a conclusion that sure it helps in 
        // the aspect of one lock but the other lock still needs to be taken.
        for(int i = (tid * m_edges) / nthreads; i < ((tid + 1) * m_edges) / nthreads; i++) {
            int node_to_move = d_csr_adj[i].src;
            int target_community_node = d_csr_adj[i].dest;

            if (node_to_move == target_community_node) continue;

            // take lock on nodes in order
            int first_node_lock = min(node_to_move, target_community_node);
            // first_node_lock = node_to_move;
            int second_node_lock = max(node_to_move, target_community_node);
            while (atomicCAS(&d_vertex_locks[first_node_lock], 0, 1) != 0) {}
            while (atomicCAS(&d_vertex_locks[second_node_lock], 0, 1) != 0) {}

            int initial_community = community[node_to_move];
            int target_community = community[target_community_node];

            if(initial_community != target_community) {
                // Take a lock on initial and target community in order of their ids to avoid deadlock
                int first_lock = min(initial_community, target_community);
                int second_lock = max(initial_community, target_community);

                // Wait until we can acquire both locks
                while (atomicCAS(&d_community_locks[first_lock], 0, 1) != 0) {}
                while (atomicCAS(&d_community_locks[second_lock], 0, 1) != 0) {}

                double delta_Q = calculate_modularity_change(n_nodes, m_edges, d_total_weight, d_csr_adj, d_csr_node_offset, community, community_degree, node_to_move, initial_community, target_community);

                #ifdef DEBUG
                printf("Node %d moving from community %d to %d gives delta Q = %f\n", node_to_move, initial_community, target_community, delta_Q);
                #endif
                if(delta_Q > 1e-12) { // Tolerance to avoid inconsistent graph reading
                    #ifdef DEBUG
                    printf("delta Q %f\n", delta_Q); //REMOVE ONLY TEMPORARY
                    #endif
                    atomicAdd((double*)&delta_Q_sum, delta_Q); //REMOVE ONLY TEMPORARY
                    community[node_to_move] = target_community;
                    update_community_degree(n_nodes, d_csr_adj, d_csr_node_offset, community, community_degree, node_to_move, initial_community, target_community);
                    atomicExch((int*)&changed, 1); //Modify later by allocating with CUDA MALLOC
                }

                // Release the locks
                atomicExch(&d_community_locks[second_lock], 0);
                atomicExch(&d_community_locks[first_lock], 0);
            }

            // Release the node locks
            atomicExch(&d_vertex_locks[second_node_lock], 0);
            atomicExch(&d_vertex_locks[first_node_lock], 0);
        }

        grid.sync();
        // Print h_community for debugging
        #ifdef DEBUG
        if(tid == 0) {
            printf("Current community assignments:\n");
            for(int i = 0; i < n_nodes; i++) {
                printf("Node %d: Community %d\n", i, community[i]);
            }
        }
        #endif
        if(tid == 0) louvain_kernel_iteration++;
        if(tid == 0) printf("An iteration completed in the loop of Louvain Kernel.\n");
        grid.sync();
    }

    if(tid == 0) {
        printf("Total delta Q in this phase: %f\n", delta_Q_sum);
    }
}

__global__ void count_communities(
    int n_nodes,
    int *d_community, 
    int *d_community_present,
    int *d_num_communities)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    
    for(int i = tid; i < n_nodes; i += nthreads) {
        int comm = d_community[i];
        atomicExch(&d_community_present[comm], 1);
    }

    __shared__ int community_count_in_block;
    if(threadIdx.x == 0) {
        community_count_in_block = 0;
    }
    __syncthreads();

    for(int i = tid; i < n_nodes; i += nthreads) {
        if(d_community_present[i] == 1) {
            atomicAdd(&community_count_in_block, 1);
        }
    }

    if(threadIdx.x == 0) atomicAdd(d_num_communities, community_count_in_block);    
}

__global__ void aggregate_graph(
    int n_nodes,
    int m_edges,
    struct Edge* d_csr_adj,
    int *d_community_present_prefixsum,
    int *d_community,
    double *d_community_degree)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    int new_n_nodes = d_community_present_prefixsum[n_nodes - 1];

    for(int i = tid; i < m_edges; i += nthreads) {
        int src = d_csr_adj[i].src;
        int dest = d_csr_adj[i].dest;
        d_csr_adj[i].src = d_community_present_prefixsum[d_community[src]] - 1;
        d_csr_adj[i].dest = d_community_present_prefixsum[d_community[dest]] - 1;
    }
    for(int i = tid; i < new_n_nodes; i += nthreads) {
        d_community_degree[i] = 0.0;
        d_community[i] = i;
    }
    grid.sync();
    for(int i = tid; i < m_edges; i += nthreads) {
        atomicAdd(&d_community_degree[d_csr_adj[i].src], d_csr_adj[i].weight);
    }
}

__global__ void copy_weights_kernel(struct Edge* out_edges, double* out_weights, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += nthreads) {
        out_edges[i].weight = out_weights[i];
    }
}

double calculate_modularity(int n_nodes, int m_edges, struct Edge* h_csr_adj, int* h_csr_node_offset, int* community) {
    double m = 0.0;
    for (int i = 0; i < m_edges; i++) {
        m += h_csr_adj[i].weight;
    }

    double Q = 0.0;
    map<int, double> community_degree;
    map<int, double> community_internal;

    for (int u = 0; u < n_nodes; u++) {
        int comm_u = community[u];
        double degree_u = 0.0;
        for (int i = h_csr_node_offset[u]; i < h_csr_node_offset[u + 1]; i++) {
            degree_u += h_csr_adj[i].weight;
            if (community[h_csr_adj[i].dest] == comm_u) {
                community_internal[comm_u] += h_csr_adj[i].weight;
            }
        }
        community_degree[comm_u] += degree_u;
    }

    for (const auto& comm_degree_pair : community_degree) {
        int comm = comm_degree_pair.first;
        double degree = comm_degree_pair.second;
        double internal = community_internal[comm];
        Q += (internal / (m)) - ((degree / (m)) * (degree / (m)));
    }

    return Q;
}

void combine_edges(struct Edge*& d_csr_adj, int& m_edges) {
    thrust::device_ptr<struct Edge> d_csr_adj_ptr = thrust::device_pointer_cast(d_csr_adj);
    thrust::sort(d_csr_adj_ptr, d_csr_adj_ptr + m_edges, EdgeComparator());
    thrust::device_ptr<struct Edge> d_csr_adj_ptr_end = d_csr_adj_ptr + m_edges;
    Edge* d_out_edges;
    double* d_out_weights;
    CUDA_CHECK(cudaMalloc(&d_out_edges, m_edges * sizeof(struct Edge)));
    CUDA_CHECK(cudaMalloc(&d_out_weights, m_edges * sizeof(double)));
    auto key_begin = thrust::make_transform_iterator(d_csr_adj_ptr, GetKey());
    auto weight_begin = thrust::make_transform_iterator(d_csr_adj_ptr, GetWeight());
    auto out_key_begin = thrust::make_transform_output_iterator(thrust::device_pointer_cast(d_out_edges), StoreKey());
    thrust::pair<decltype(out_key_begin), thrust::device_ptr<double>> result =
        thrust::reduce_by_key(
            key_begin, key_begin + m_edges,  // input keys
            weight_begin,                    // input values
            out_key_begin,                   // output keys
            thrust::device_pointer_cast(d_out_weights),  // output values
            thrust::equal_to<thrust::pair<int,int>>(),
            thrust::plus<double>()
        );
    int new_m_edges = result.first - out_key_begin;
    // cudaMemcpy(h_csr_adj, d_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyDeviceToHost);
    cout << "New number of edges after aggregation: " << new_m_edges << endl;
    thrust::device_ptr<double> w = thrust::device_pointer_cast(d_out_weights);
    copy_weights_kernel<<<1, 1024>>>(d_out_edges, d_out_weights, new_m_edges);
    cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaFree(d_csr_adj));
    d_csr_adj = d_out_edges;
    m_edges = new_m_edges;     // Update edge count
    cudaFree(d_out_weights);
}


void louvain_cuda(vector<vector<pair<int, double>>> &adj, int n_nodes, int m_edges) {
    int n = adj.size();

    // Create CSR representation
    struct Edge* h_csr_adj = new struct Edge[m_edges];
    int* h_csr_node_offset = new int[n_nodes + 1];
    int* h_community = new int[n_nodes];
    double* h_total_weight = new double; *h_total_weight = 0.0; // actually double total weight
    double* h_community_degree = new double[n_nodes];
    for (int i = 0; i < n_nodes; i++) {
        h_community_degree[i] = 0.0;
    }

    int edge_counter = 0;
    for (int i = 0; i < n_nodes; i++) {
        h_csr_node_offset[i] = edge_counter;
        h_community[i] = i;
        for (const auto &edge : adj[i]) {
            h_csr_adj[edge_counter].src = i;
            h_csr_adj[edge_counter].dest = edge.first;
            h_csr_adj[edge_counter].weight = edge.second;
            edge_counter++;
            h_community_degree[i] += edge.second;
            h_total_weight[0] += edge.second;
        }
    }
    h_csr_node_offset[n_nodes] = m_edges;

    // Allocate device memory
    struct Edge* d_csr_adj;
    int* d_csr_node_offset;
    int *d_community;
    int *d_community_locks, *d_vertex_locks;
    double *d_total_weight, *d_community_degree;

    cudaMalloc(&d_csr_adj, m_edges * sizeof(struct Edge));
    cudaMalloc(&d_csr_node_offset, (n_nodes + 1) * sizeof(int));
    cudaMalloc(&d_community, n_nodes * sizeof(int));
    cudaMalloc(&d_community_degree, n_nodes * sizeof(double));
    cudaMalloc(&d_vertex_locks, n_nodes * sizeof(int));
    cudaMemset(d_vertex_locks, 0, n_nodes * sizeof(int));
    cudaMalloc(&d_community_locks, n_nodes * sizeof(int));
    cudaMemset(d_community_locks, 0, n_nodes * sizeof(int));
    cudaMalloc(&d_total_weight, sizeof(double));

    cudaMemcpy(d_csr_adj, h_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_node_offset, h_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, h_community, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_degree, h_community_degree, n_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_weight, h_total_weight, sizeof(double), cudaMemcpyHostToDevice);

    // d_community_present_prefixsum to help in aggregation
    // It contains 1 for communities present and 0 for not present after the count_communities kernel
    // It is prefix summed to get new community indexes, this is done in the aggregate_graph kernel
    int* d_community_present_prefixsum, *d_num_communities;
    cudaMalloc(&d_community_present_prefixsum, n_nodes * sizeof(int));
    cudaMalloc(&d_num_communities, sizeof(int));

    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community) << " is the current modularity." << endl;

    // Time this with chrono
    auto start = std::chrono::steady_clock::now();    
    int old_n_nodes = -1;
    while(old_n_nodes != n_nodes) {
    
    old_n_nodes = n_nodes;
    // Run few times to converge
    #ifdef DEBUG
    for (int i = 0; i < m_edges; i++) {
        cout << "Edge " << i << ": " << h_csr_adj[i].src << " -> " << h_csr_adj[i].dest << " with weight " << h_csr_adj[i].weight << endl;
    }
    for (int i = 0; i < n_nodes + 1; i++) {
        cout << "Node offset " << i << ": " << h_csr_node_offset[i] << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
        cout << "Community degree " << i << ": " << h_community_degree[i] << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
        cout << "Partition " << i << " is in community " << h_community[i] << endl;
    }
    #endif

    // Cooperative Launch
    void* kernelArgs[] = {
        (void*)&n_nodes,
        (void*)&m_edges,
        (void*)&d_csr_adj,
        (void*)&d_csr_node_offset,
        (void*)&d_community,
        (void*)&d_community_degree,
        (void*)&d_vertex_locks,
        (void*)&d_community_locks,
        (void*)&d_total_weight
    };
    cudaLaunchCooperativeKernel((void*)louvain_kernel, 32, 512, kernelArgs);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cout << "Louvain kernel phase completed." << endl;

    cudaMemset(d_community_present_prefixsum, 0, n_nodes * sizeof(int));
    cudaMemset(d_num_communities, 0, sizeof(int));
    count_communities<<<1, 1024>>>(
        n_nodes, 
        d_community, 
        d_community_present_prefixsum, 
        d_num_communities);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_num_communities;
    cudaMemcpy(&h_num_communities, d_num_communities, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "New Number of communities: " << h_num_communities << endl;

    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_community_present_prefixsum);
    thrust::inclusive_scan(dev_ptr, dev_ptr + n_nodes, dev_ptr);
    #ifdef DEBUG
    for(int i = 0; i < n_nodes; i++) {
        int temp;
        cudaMemcpy(&temp, d_community_present_prefixsum + i, sizeof(int), cudaMemcpyDeviceToHost);
        cout << "Community presence at (Prefix Summed)" << i << " is " << temp << endl;
    }
    #endif

    // Output the communities
    cudaMemcpy(h_community, d_community, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    #ifdef DEBUG
    for (int i = 0; i < n_nodes; i++) {
        cout << "Partition " << i << " is in community " << h_community[i] << endl;
    }
    #endif

    // Aggregate Graph:
    // Change edges based on the new community indexes
    // Change the community array = i cuz aggr is done. New partitions.
    // Set the community degree based on the new indexes
    void *kernelArgs1[] = {
        (void*)&n_nodes,
        (void*)&m_edges,
        (void*)&d_csr_adj,
        (void*)&d_community_present_prefixsum,
        (void*)&d_community,
        (void*)&d_community_degree
    };
    cudaLaunchCooperativeKernel((void*)aggregate_graph, 32, 512, kernelArgs1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    n_nodes = h_num_communities;
    // Copy data from device to host
    cudaMemcpy(h_csr_adj, d_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csr_node_offset, d_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community, d_community, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community_degree, d_community_degree, n_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    
    #ifdef DEBUG
    // Output the communities
    for (int i = 0; i < n_nodes; i++) {
        cout << "Partition " << i << " is in community " << h_community[i] << endl;
    }
    for (int i = 0; i < m_edges; i++) {
        cout << "Edge " << i << ": " << h_csr_adj[i].src << " -> " << h_csr_adj[i].dest << " with weight " << h_csr_adj[i].weight << endl;
    }
    #endif

    // Need to set node_offsets for new adjacency lists
    // Also, need to aggregate the edges with same src and dest
    // Sort the adj list to make it CSR format again, use thrust
    combine_edges(d_csr_adj, m_edges);
    cudaMemcpy(h_csr_adj, d_csr_adj, m_edges * sizeof(struct Edge), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community, d_community, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community_degree, d_community_degree, n_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    #ifdef DEBUG
    for (int i = 0; i < m_edges; i++) {
        cout << "After combining edges, Edge " << i << ": " << h_csr_adj[i].src << " -> " << h_csr_adj[i].dest << " with weight " << h_csr_adj[i].weight << endl;
    }
    #endif

    // Set node offsets
    thrust::device_ptr<struct Edge> d_csr_adj_ptr = thrust::device_pointer_cast(d_csr_adj);
    thrust::device_ptr<struct Edge> d_csr_adj_ptr_end = d_csr_adj_ptr + m_edges;
    cudaMemset(d_csr_node_offset, 0, (n_nodes + 1) * sizeof(int));
    thrust::for_each(d_csr_adj_ptr, d_csr_adj_ptr_end, CountEdgesBySrc(d_csr_node_offset));
    // Prefix sum to get correct offsets
    thrust::device_ptr<int> dev_ptr_offsets = thrust::device_pointer_cast(d_csr_node_offset);
    thrust::exclusive_scan(dev_ptr_offsets, dev_ptr_offsets + n_nodes + 1, dev_ptr_offsets);

    cudaMemcpy(h_csr_node_offset, d_csr_node_offset, (n_nodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community) << " is the current modularity." << endl;
    }
    auto end = std::chrono::steady_clock::now();

    cout << "Final number of nodes: " << n_nodes << " and edges: " << m_edges << endl;
    cout << calculate_modularity(n_nodes, m_edges, h_csr_adj, h_csr_node_offset, h_community) << " is the final modularity." << endl;
    cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << endl;

    // Free host memory
    delete[] h_csr_adj;
    delete[] h_csr_node_offset;
    delete[] h_community;
    delete[] h_community_degree;
}

int main() {
    int n_nodes, m_edges;
    cin >> n_nodes >> m_edges;
    vector<vector<pair<int, double>>> adj(n_nodes);
    for (int i = 0; i < m_edges; i++) {
        int u, v;
        double w;
        cin >> u >> v >> w;
        // w = 1.0;
        // Imagine using emplace back instead of push back
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }

    louvain_cuda(adj, n_nodes, m_edges * 2);
    return 0;
}