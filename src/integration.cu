#include "integration.h"
#include "langermann.cuh"

#include <cmath>
#include <limits>

__device__ size_t xy_to_thread_idx(size_t x, size_t y) {
    return y * blockDim.x + x;
}

__global__ void sum_in_each_block(const double *arr, size_t len, double *out_arr) {
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ double shared_arr[];
    shared_arr[threadIdx.x] = (global_thread_idx < len ? arr[global_thread_idx] : 0);
    size_t prev_num_sum = blockDim.x;
    size_t num_sum;
    do {
        num_sum = (prev_num_sum + 1) / 2;
        __syncthreads();
        if (threadIdx.x < num_sum) {
            if (num_sum + threadIdx.x < prev_num_sum) {
                shared_arr[threadIdx.x] += shared_arr[num_sum + threadIdx.x];
            }
        }
        prev_num_sum = num_sum;
    } while (num_sum != 1);
    if (threadIdx.x == 0) {
        out_arr[blockIdx.x] = shared_arr[0];
    }
}

__global__ void calc_integral(double x_start, double y_start, double delta_x, double delta_y,
                              int num_steps_x, int num_steps_y, double *out_sum) {
    extern __shared__ double shared_res[];
    size_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    double value_at_point;
    if (x_idx >= num_steps_x || y_idx >= num_steps_y) {
        value_at_point = 0;
    } else {
        value_at_point = langermann_function(x_start + x_idx * delta_x, y_start + y_idx * delta_y);
    }
    size_t shared_res_idx = threadIdx.y * blockDim.x + threadIdx.x;
    shared_res[shared_res_idx] = value_at_point;
    __syncthreads();
    if (threadIdx.y == 0) {
        for (size_t y = 1; y < blockDim.y; ++y) {
            shared_res[xy_to_thread_idx(threadIdx.x, 0)] += shared_res[xy_to_thread_idx(threadIdx.x, y)];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t x = 1; x < blockDim.x; ++x) {
            shared_res[0] += shared_res[xy_to_thread_idx(x, 0)];
        }
        out_sum[gridDim.x * blockIdx.y + blockIdx.x] = shared_res[0];
    }
}

double sum_array_gpu(double *arr, size_t len) {
    while (len != 1) {
        size_t new_len = (len + 1023) / 1024;
        sum_in_each_block<<<new_len, 1024, sizeof(double) * 1024>>>(arr,
                                                                    len,
                                                                    arr);
        len = new_len;
    }
    double res[1];
    cudaMemcpy(res, arr, sizeof(double), cudaMemcpyDeviceToHost);
    return res[0];
}

double calc_integral_on_gpu(double x_start, double x_end, double y_start, double y_end,
                            int num_steps_x, int num_steps_y) {
    double delta_x = (x_end - x_start) / num_steps_x;
    double delta_y = (y_end - y_start) / num_steps_y;

    double *buffer;
    dim3 threads_per_block{32, 32};
    dim3 num_blocks((num_steps_x + threads_per_block.x - 1) / threads_per_block.x,
                    (num_steps_y + threads_per_block.y - 1) / threads_per_block.y);
    cudaMalloc(&buffer, sizeof(double) * num_blocks.x * num_blocks.y);
    size_t num_threads_per_block = threads_per_block.x * threads_per_block.y;
    calc_integral<<<num_blocks, threads_per_block, sizeof(double) * num_threads_per_block>>>(
            x_start, y_start, delta_x, delta_y, num_steps_x, num_steps_y, buffer);

    double summ = sum_array_gpu(buffer, num_blocks.x * num_blocks.y);

    cudaFree(buffer);
    return summ * delta_x * delta_y;
}

void cuda_integration(const Config &conf,
                      double &outCalculatedIntegral, double &outAbsError, double &outRelError) {
    outAbsError = std::numeric_limits<double>::infinity();
    outRelError = std::numeric_limits<double>::infinity();
    outCalculatedIntegral = calc_integral_on_gpu(conf.x_start, conf.x_end,
                                                 conf.y_start, conf.y_end,
                                                 conf.init_steps_x, conf.init_steps_y);
#ifdef DEBUG_OUTPUT
    std::cout << "Integral after iteration #" << 1 << " = " << outCalculatedIntegral << std::endl;
#endif

    // Approximating the value of integral by increasing number of partitions
    int steps_x = conf.init_steps_x * 2;
    int steps_y = conf.init_steps_y * 2;
    for (int n_iter = 1; n_iter < conf.max_iter && !(outAbsError < conf.abs_err && outRelError < conf.rel_err); ++n_iter) {
        double integral_old = outCalculatedIntegral;
        double integral_new = calc_integral_on_gpu(conf.x_start, conf.x_end,
                                                          conf.y_start, conf.y_end,
                                                          steps_x, steps_y);

        outAbsError = std::abs(integral_new - integral_old);

        if (integral_old == 0.0) { // division by 0.0 is UB
            outRelError = std::numeric_limits<double>::infinity();
        } else {
            outRelError = outAbsError / std::abs(integral_old);
        }

        steps_x *= 2;
        steps_y *= 2;
        outCalculatedIntegral = integral_new;

#ifdef DEBUG_OUTPUT
        std::cout << "Integral after iteration #" << n_iter + 1 << " = " << outCalculatedIntegral << std::endl;
#endif
    }
}