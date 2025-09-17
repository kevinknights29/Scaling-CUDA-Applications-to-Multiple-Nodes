#include <iostream>
#include <limits>
#include <cstdio>
#include <cmath>

#include <nvshmem.h>
#include <nvshmemx.h>

inline void CUDA_CHECK (cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// Number of points in the overall spatial domain
#define NUM_POINTS 1048576

__global__ void wave_update (float* u, const float* u_old, const float* u_older, float dtdxsq, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    bool on_boundary = false;
    if (my_pe == 0 && idx == 0) {
        on_boundary = true;
    }
    else if (my_pe == n_pes - 1 && idx == N-1) {
        on_boundary = true;
    }

    if (idx < N && !on_boundary) {
        float u_old_left;

        if (idx == 0) {
            // Note we do not get here if we're PE == 0
            u_old_left = nvshmem_float_g(&u_old[N-1], my_pe - 1);
        }
        else {
            u_old_left = u_old[idx-1];
        }

        float u_old_right;
        if (idx == N-1) {
            // Note we do not get here if we're PE == n_pes - 1
            u_old_right = nvshmem_float_g(&u_old[0], my_pe + 1);
        }
        else {
            u_old_right = u_old[idx+1];
        }

        u[idx] = 2.0f * u_old[idx] - u_older[idx] +
                 dtdxsq * (u_old_right - 2.0f * u_old[idx] + u_old_left);
    }
}

__global__ void initialize (float* u, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int offset = nvshmem_my_pe() * (NUM_POINTS / nvshmem_n_pes());

    if (idx < N) {
        u[idx] = std::sin(2.0f * M_PI * (idx + offset) / static_cast<float>(NUM_POINTS - 1));
    }
}

__global__ void check_solution (float* u, float* l2_norm, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int offset = nvshmem_my_pe() * (NUM_POINTS / nvshmem_n_pes());

    if (idx < N) {
        float u_correct = std::sin(2.0f * M_PI * (idx + offset) / static_cast<float>(NUM_POINTS - 1));
        float l2 = (u[idx] - u_correct) * (u[idx] - u_correct);
        atomicAdd(l2_norm, l2);
    }
}

int main() {
    // Initialize NVSHMEM
    nvshmem_init();

    // Obtain our NVSHMEM processing element ID and number of PEs
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Each PE (arbitrarily) chooses the GPU corresponding to its ID
    int device = my_pe;
    CUDA_CHECK(cudaSetDevice(device));

    const int N = NUM_POINTS / n_pes;

    // Allocate space for the grid data, and the temporary buffer
    // for the "old" and "older" data.
    float* u_older = (float*) nvshmem_malloc(N * sizeof(float));
    float* u_old = (float*) nvshmem_malloc(N * sizeof(float));
    float* u = (float*) nvshmem_malloc(N * sizeof(float));

    // Initialize the data
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    initialize<<<blocks, threads_per_block>>>(u_older, N);
    initialize<<<blocks, threads_per_block>>>(u_old, N);
    initialize<<<blocks, threads_per_block>>>(u, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now iterate until we've completed a full period
    const float period = 1.0f;
    const float start_time = 0.0f;
    const float stop_time = period;

    // Maximum stable timestep is <= dx
    float stability_factor = 0.5f;
    float dx = 1.0f / (NUM_POINTS - 1);
    float dt = stability_factor * dx;

    float t = start_time;
    const float safety_factor = (1.0f - 1.0e-5f);

    int num_steps = 0;

    while (t < safety_factor * stop_time) {
        // Make sure the last step does not go over the target time
        if (t + dt >= stop_time) {
            dt = stop_time - t;
        }

        float dtdxsq = (dt / dx) * (dt / dx);

        // Launch kernel to do the calculation
        wave_update<<<blocks, threads_per_block>>>(u, u_old, u_older, dtdxsq, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Keep PEs synchronized
        nvshmem_barrier_all();

        // Swap u_old and u_older
        std::swap(u_old, u_older);

        // Swap u and u_old
        std::swap(u, u_old);

        // Print out diagnostics periodically
        if (num_steps % 100000 == 0 && my_pe == 0) {
            std::cout << "Current integration time = " << t << "\n";
        }

        // Update t
        t += dt;
        ++num_steps;
    }

    // Check how close we are to the initial configuration
    float* l2_norm = (float*) malloc(sizeof(float));
    float* d_l2_norm = (float*) nvshmem_malloc(sizeof(float));
    CUDA_CHECK(cudaMemset(d_l2_norm, 0, sizeof(float)));

    check_solution<<<blocks, threads_per_block>>>(u, d_l2_norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce in place over all PEs
    nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, d_l2_norm, d_l2_norm, 1);

    CUDA_CHECK(cudaMemcpy(l2_norm, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost));

    // Normalize by number of grid points and take square root
    *l2_norm = std::sqrt(*l2_norm / NUM_POINTS);

    if (my_pe == 0) {
        std::cout << "Error = " << *l2_norm << "\n";
    }

    // Clean up
    nvshmem_free(u_older);
    nvshmem_free(u_old);
    nvshmem_free(u);

    // Finalize NVSHMEM
    nvshmem_finalize();

    return 0;
}
