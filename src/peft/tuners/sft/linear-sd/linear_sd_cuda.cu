#include <cassert>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024


template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ input,
    const int64_t input_row_stride,
    const int64_t input_col_stride,
    const scalar_t* __restrict__ output_grad,
    const int64_t output_grad_row_stride,
    const int64_t output_grad_col_stride,
    const int64_t* __restrict__ input_indices,
    const int64_t* __restrict__ output_indices,
    const int64_t N,
    const int64_t batch_d,
    const int64_t input_d,
    const int64_t output_d,
    float* __restrict__ dv_grad
) {
    const int64_t n = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (n >= N)
        return;
    const int64_t input_index = input_indices[n];
    const int64_t output_index = output_indices[n];
    int64_t input_begin = input_index * input_col_stride;
    int64_t output_begin = output_index * output_grad_col_stride;

    float result = 0.0;
    for (int64_t i = 0; i < batch_d; i++)
        result += input[input_begin + i * input_row_stride] * output_grad[output_begin + i * output_grad_row_stride];

    dv_grad[n] = result;
}


torch::Tensor linear_sd_cuda_backward(
    torch::Tensor input,
    torch::Tensor output_grad,
    torch::Tensor di
) {
    assert(input.layout() == torch::kStrided);
    assert(output_grad.layout() == torch::kStrided);
    assert(input.dim() == 2);
    assert(output_grad.dim() == 2);
    assert(di.dim() == 1);
    const int64_t N = di.size(0);
    torch::Tensor dv_grad = torch::zeros(
        N, 
        torch::TensorOptions().dtype(torch::kFloat32).device(output_grad.device())
    );
    //torch::Tensor output_indices = di.floor_divide(input.size(0)); //di / input.size(1);
    //assert(output_indices.dtype() == torch::kInt64);
    //torch::Tensor input_indices = di - input.size(0) * output_indices;
    torch::Tensor rows = di.floor_divide(input.size(1));
    torch::Tensor columns = di - input.size(1) * rows;

    const dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::ScalarType::Half,
        torch::ScalarType::BFloat16,
        input.type(),
        "linear_sd_cuda_backward",
        ([&] {
            linear_sd_backward_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                input.data<scalar_t>(),
                input.stride(0),
                input.stride(1),
                output_grad.data<scalar_t>(),
                output_grad.stride(0),
                output_grad.stride(1),
                columns.data<int64_t>(),
                rows.data<int64_t>(),
                N,
                input.size(0),
                input.size(1),
                output_grad.size(1),
                dv_grad.data<float>()
            );
        })
    );

    return dv_grad;
}


