#include <cassert>
//#define USE_CUDA_DSA 1

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using index_t = int32_t;
const torch::Dtype index_dtype = torch::kInt32;

const index_t ELEMENT_RANGES_BLOCKSIZE = 1024;

__global__ void element_ranges_kernel(
    const index_t* __restrict__ A,
    const index_t N,
    index_t* __restrict__ begins,
    index_t* __restrict__ ends
) {
    __shared__ index_t values[ELEMENT_RANGES_BLOCKSIZE];
    
    const index_t idx = threadIdx.x + (blockDim.x - 1) * blockIdx.x;
    index_t v;
    const bool thread_active = idx < N;
    if (thread_active) {
        v = A[idx];
        values[threadIdx.x] = v;
    }
    __syncthreads();
    
    if (thread_active) {
        if (idx == 0)
            begins[v] = 0;

        if (threadIdx.x != 0) {
            const index_t prev = values[threadIdx.x - 1];
            if (v != prev) {
                begins[v] = idx;
                ends[prev] = idx;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> element_ranges(
    torch::Tensor sorted_values,
    const int64_t ub
) {
    assert(sorted_values.dim() == 1);
    const index_t N = sorted_values.size(0);
    torch::Tensor begins = torch::full(
        ub, N,
        torch::TensorOptions().dtype(index_dtype).device(sorted_values.device())
    );
    torch::Tensor ends = torch::full(
        ub, N,
        torch::TensorOptions().dtype(index_dtype).device(sorted_values.device())
    );
    index_t num_blocks = (N + ELEMENT_RANGES_BLOCKSIZE - 2) / (ELEMENT_RANGES_BLOCKSIZE - 1);
    element_ranges_kernel<<<dim3(num_blocks), dim3(ELEMENT_RANGES_BLOCKSIZE)>>>(
        sorted_values.data<index_t>(),
        index_t(N),
        begins.data<index_t>(),
        ends.data<index_t>()
    );
    return {begins, ends};
}

#define DTYPE_PATCH_WA(dtype_bytes) 512
#define DTYPE_PATCH_WB(dtype_bytes) 384
#define DTYPE_PATCH_H(dtype_bytes) 8

const index_t RESULTS_PER_BLOCK = 32;
const index_t THREADS_PER_BLOCK = 32;

template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const index_t* __restrict__ Ai,
    const index_t* __restrict__ Bi,
    const index_t N,
    const index_t Ad,
    const index_t Bd,
    const index_t h,
    float* __restrict__ outputs
) {
    __shared__ float partial_sums[RESULTS_PER_BLOCK * THREADS_PER_BLOCK];

    const index_t n = threadIdx.x / THREADS_PER_BLOCK + blockIdx.x * RESULTS_PER_BLOCK;
    if (n < N) {
        const index_t warp_idx = threadIdx.x % THREADS_PER_BLOCK;
        const index_t Abegin = Ai[n] * h;
        const index_t Bbegin = Bi[n] * h;
        float sum = 0.0;
        
        for (index_t i = warp_idx; i < h; i += THREADS_PER_BLOCK)
            sum += A[i + Abegin] * B[i + Bbegin];

        partial_sums[threadIdx.x] = sum;
        for (index_t i = 1; i < THREADS_PER_BLOCK && ! (i & warp_idx); i <<= 1)
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + i];

        if (warp_idx == 0)
            outputs[n] = partial_sums[threadIdx.x];
    }
}

//template <>
//__global__ void linear_sd_backward_kernel(
//    const nv_bfloat16* __restrict__ A,
//    const nv_bfloat16* __restrict__ B,
//    const index_t* __restrict__ Ai,
//    const index_t* __restrict__ Bi,
//    const index_t N,
//    const index_t Ad,
//    const index_t Bd,
//    const index_t h,
//    float* __restrict__ outputs
//) {
//    __shared__ float partial_sums[RESULTS_PER_BLOCK * (THREADS_PER_RESULT + 1)];
//
//    const index_t n = threadIdx.x / THREADS_PER_RESULT + blockIdx.x * RESULTS_PER_BLOCK;
//    if (n < N) {
//        const index_t warp_idx = threadIdx.x % THREADS_PER_RESULT;
//        const index_t Abegin = Ai[n] * h;
//        const index_t Bbegin = Bi[n] * h;
//        float sum = 0.0;
//        
//        for (index_t i = 2 * warp_idx; i < h; i += 2 * THREADS_PER_RESULT) {
//            const nv_bfloat162 Aval = *reinterpret_cast<const nv_bfloat162*>(&A[i + Abegin]);
//            const nv_bfloat162 Bval = *reinterpret_cast<const nv_bfloat162*>(&B[i + Bbegin]);
//            const nv_bfloat162 prod = Aval * Bval;
//            const float2 fprod = __bfloat1622float2(prod);
//            sum += fprod.x + fprod.y;
//            //sum += __bfloat162float(A[i + Abegin] * B[i + Bbegin]);
//        }
//
//        const index_t Sbegin = warp_idx + (THREADS_PER_RESULT + 1) * (threadIdx.x / THREADS_PER_RESULT);
//        partial_sums[Sbegin] = sum;
//        for (index_t i = 1; i < THREADS_PER_RESULT && ! (i & warp_idx); i <<= 1)
//            partial_sums[Sbegin] += partial_sums[Sbegin + i];
//
//        if (warp_idx == 0)
//            outputs[n] = partial_sums[Sbegin];
//    }
//}



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

    di = di.to(index_dtype);
    const index_t N = di.size(0);
    //torch::Tensor output_indices = di.floor_divide(input.size(0)); //di / input.size(1);
    //assert(output_indices.dtype() == torch::kInt64);
    //torch::Tensor input_indices = di - input.size(0) * output_indices;
    const index_t Ad = input.size(0);
    const index_t Bd = output_grad.size(0);
    const index_t h = input.size(1);
    
    //auto sorted_indices = torch::sort(di);
    //di = std::get<0>(sorted_indices);
    //torch::Tensor perm = std::get<1>(sorted_indices);
    //torch::Tensor inverse_perm = torch::empty_like(perm);
    //inverse_perm.index_put_(
    //    {perm},
    //    torch::arange(
    //        N,
    //        torch::TensorOptions()
    //            .dtype(perm.dtype())
    //            .device(perm.device())
    //    )
    //);

    torch::Tensor Ai = torch::remainder(di, Ad);
    torch::Tensor Bi = di.floor_divide(Ad);

    //auto Ai_ranges = element_ranges(Ai, Ad);
    //torch::Tensor Bi_per_Ai = std::get<1>(Ai_ranges) - std::get<0>(Ai_ranges);
    //Bi_per_Ai.add_(RESULTS_PER_BLOCK - 1).div_(RESULTS_PER_BLOCK);
    //torch::Tensor Arow_ends = torch::cumsum(Bi_per_Ai, 0);
    //const index_t n_blocks = Bi_per_Ai.sum();
    //torch::Tensor Arows = torch::searchsorted(
    //    Arow_ends,
    //    torch::arange(
    //        n_blocks,
    //        torch::TensorOptions()
    //            .dtype(Arow_ends.dtype())
    //            .device(Arow_ends.device())
    //    )
    //)
    //torch::Tensor Bbegins = 
    //const index_t dtype_bytes = torch::elementSize(torch::typeMetaToScalarType(input.dtype()));
    //const index_t PATCH_WA = DTYPE_PATCH_WA(dtype_bytes);
    //const index_t PATCH_WB = DTYPE_PATCH_WB(dtype_bytes);
    //const index_t PATCH_H = DTYPE_PATCH_H(dtype_bytes);

    //const index_t An = (Ad + PATCH_WA - 1) / PATCH_WA;
    //const index_t Bn = (Bd + PATCH_WB - 1) / PATCH_WB;

    const dim3 grid((N + RESULTS_PER_BLOCK - 1) / RESULTS_PER_BLOCK);

    torch::Tensor result = torch::zeros(
        N, 
        torch::TensorOptions().dtype(torch::kFloat32).device(output_grad.device())
    );

    //assert(input.scalar_type() == torch::ScalarType::BFloat16);
    //linear_sd_backward_kernel<<<grid, 1024>>>(
    //    (nv_bfloat16*)input.data<torch::BFloat16>(),
    //    (nv_bfloat16*)output_grad.data<torch::BFloat16>(),
    //    Ai.data<index_t>(),
    //    Bi.data<index_t>(),
    //    N,
    //    Ad,
    //    Bd,
    //    h,
    //    result.data<float>()
    //);


    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::ScalarType::Half,
        torch::ScalarType::BFloat16,
        input.type(),
        "linear_sd_cuda_backward",
        ([&] {
            linear_sd_backward_kernel<scalar_t><<<grid, RESULTS_PER_BLOCK * THREADS_PER_BLOCK>>>(
                input.data<scalar_t>(),
                output_grad.data<scalar_t>(),
                Ai.data<index_t>(),
                Bi.data<index_t>(),
                N,
                Ad,
                Bd,
                h,
                result.data<float>()
            );
        })
    );

    return result; //.index({inverse_perm});
}


