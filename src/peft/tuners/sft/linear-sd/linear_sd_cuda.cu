#include <cassert>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_DIM 1024
#define WARP_DIM 32
#define BLOCK_SIZE (THREAD_DIM / WARP_DIM)


template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output_grad,
    const int64_t* __restrict__ input_indices,
    const int64_t* __restrict__ output_indices,
    const int64_t N,
    const int64_t batch_d,
    const int64_t input_d,
    const int64_t output_d,
    float* __restrict__ dv_grad
) {
    const int64_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x / WARP_DIM;
    if (n >= N)
        return;
    const int64_t w = threadIdx.x % WARP_DIM;
    const int64_t input_index = input_indices[n];
    const int64_t output_index = output_indices[n];
    __shared__ float warp_results[BLOCK_SIZE][WARP_DIM];

    const scalar_t* input_row = input + input_index * batch_d; 
    const scalar_t* output_row = output_grad + output_index * batch_d; 
    float result = 0.0;
    for (int64_t i = w; i < batch_d; i += WARP_DIM) {
        result += float(input_row[i]) * float(output_row[i]);
        __syncthreads();
    }
        //result += output_grad[i * output_d + row] * input[i * input_d + col];
    //assert(result != 0.0);
    warp_results[threadIdx.x / WARP_DIM][w] = result;
    __syncthreads();

    if (w == 0) {
        result = 0.0;
        for (int64_t i = 0; i < WARP_DIM; i++)
            result += warp_results[threadIdx.x / WARP_DIM][i];
        //assert(result != 0.0);
        dv_grad[n] = result;
    }
}


torch::Tensor linear_sd_cuda_backward(
    torch::Tensor input,
    torch::Tensor output_grad,
    torch::Tensor di
) {
    assert(input.dim() == 2);
    assert(input.stride(0) == input.size(1));
    assert(input.stride(1) == 1);
    assert(output_grad.dim() == 2);
    assert(output_grad.stride(0) == output_grad.size(1));
    assert(output_grad.stride(1) == 1);
    const int64_t N = di.size(0);
    torch::Tensor dv_grad = torch::zeros(
        N, 
        torch::TensorOptions().dtype(torch::kFloat32).device(output_grad.device())
    );
    torch::Tensor output_indices = di.floor_divide(input.size(0)); //di / input.size(1);
    assert(output_indices.dtype() == torch::kInt64);
    torch::Tensor input_indices = di - input.size(0) * output_indices;

    const dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::ScalarType::Half,
        torch::ScalarType::BFloat16,
        input.type(),
        "linear_sd_cuda_backward",
        ([&] {
            linear_sd_backward_kernel<scalar_t><<<blocks, THREAD_DIM>>>(
                input.data<scalar_t>(),
                output_grad.data<scalar_t>(),
                input_indices.data<int64_t>(),
                output_indices.data<int64_t>(),
                N,
                input.size(1),
                input.size(0),
                output_grad.size(0),
                dv_grad.data<float>()
            );
        })
    );

    return dv_grad;
}


