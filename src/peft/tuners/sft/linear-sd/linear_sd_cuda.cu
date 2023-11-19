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

const index_t PATCH_WA = 512;
const index_t PATCH_WB = 196;
const index_t PATCH_H = 16;

template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const index_t* __restrict__ Ai,
    const index_t* __restrict__ Bi,
    const index_t* __restrict__ pair_begins,
    const index_t* __restrict__ pair_ends,
    const index_t Ad,
    const index_t Bd,
    const index_t h,
    float* __restrict__ outputs
) {
    __shared__ float cached_A[PATCH_H * PATCH_WA];
    __shared__ float cached_B[PATCH_H * PATCH_WB];

    const index_t Ax = threadIdx.x + PATCH_WA * blockIdx.x;
    const index_t ybegin = blockIdx.y * PATCH_H;
    const index_t yend = std::min(ybegin + PATCH_H, h);
    const index_t yrange = yend - ybegin;

    if (Ax < Ad) {
        for (index_t i = 0; i < yrange; i++)
            cached_A[threadIdx.x + i * PATCH_WA] = A[Ax + (i + ybegin) * Ad];
    }

    for (index_t Bx = 0; Bx < Bd; Bx += PATCH_WB) {
        __syncthreads();
        for (index_t i = threadIdx.x; i < PATCH_H * PATCH_WB; i += PATCH_WA) {
            const index_t Bcol = Bx + i % PATCH_WB;
            const index_t Brow = ybegin + i / PATCH_WB;
            if (Bcol < Bd && Brow < h)
                cached_B[i] = B[Bcol + Brow * Bd];
        }
        __syncthreads();

        const index_t pair_id = blockIdx.x + gridDim.x * (Bx / PATCH_WB);
        //assert(pair_id < N_pairs);
        const index_t pair_begin = pair_begins[pair_id];
        const index_t pair_end = pair_ends[pair_id];
        for (index_t k = pair_begin + threadIdx.x; k < pair_end; k += PATCH_WA) {
            //assert(Ai[k] / PATCH_WA == blockIdx.x);
            //assert(k < N_indices);
            const index_t Api = Ai[k] - PATCH_WA * blockIdx.x;
            //assert(Api >= 0);
            //assert(Api < PATCH_WA);
            const index_t Bpi = Bi[k] - Bx;
            //assert(Bpi >= 0);
            //assert(Bpi < PATCH_WB);
            float sum = 0.0;
            for (index_t i = 0; i < yrange; i++)
                sum += float(cached_A[Api + i * PATCH_WA]) * float(cached_B[Bpi + i * PATCH_WB]);
            atomicAdd(&outputs[k], sum);
        }
    }
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

    di = di.to(index_dtype);
    const index_t N = di.size(0);
    //torch::Tensor output_indices = di.floor_divide(input.size(0)); //di / input.size(1);
    //assert(output_indices.dtype() == torch::kInt64);
    //torch::Tensor input_indices = di - input.size(0) * output_indices;
    const index_t Ad = input.size(1);
    const index_t Bd = output_grad.size(1);
    const index_t h = input.size(0);
    const index_t An = (Ad + PATCH_WA - 1) / PATCH_WA;
    const index_t Bn = (Bd + PATCH_WB - 1) / PATCH_WB;

    const dim3 grid(An, (h + PATCH_H - 1) / PATCH_H);

    torch::Tensor Ai = torch::remainder(di, Ad);
    //assert(torch::all(Ai < Ad).item<bool>());
    torch::Tensor Bi = di.floor_divide(Ad);
    //assert(torch::all(Bi < Bd).item<bool>());
    torch::Tensor Apx = Ai.floor_divide(PATCH_WA);
    //assert(torch::all(Apx < An).item<bool>());
    torch::Tensor Bpx = Bi.floor_divide(PATCH_WB);
    //assert(torch::all(Bpx < Bn).item<bool>());
    torch::Tensor pair_ids = Apx + An * Bpx;
    auto sorted_pairs = torch::sort(pair_ids);
    pair_ids = std::get<0>(sorted_pairs);
    torch::Tensor pair_perm = std::get<1>(sorted_pairs);
    torch::Tensor inverse_perm = torch::empty_like(pair_perm);
    inverse_perm.index_put_(
        {pair_perm}, 
        torch::arange(
            N,
            torch::TensorOptions()
                .dtype(pair_perm.dtype())
                .device(pair_perm.device())
        )
    );
    //pair_ids = pair_ids.index({pair_perm});
    //assert(pair_ids.layout() == torch::kStrided && pair_ids.stride(0) == 1);
    //assert(torch::all(
    //    pair_ids.index({torch::indexing::Slice(0, -1)}) <=
    //    pair_ids.index({torch::indexing::Slice(1, torch::indexing::None)})
    //).item<bool>());
    Ai = Ai.index({pair_perm});
    Bi = Bi.index({pair_perm});
    
    const index_t n_pairs = An * Bn;
    auto pair_ranges = element_ranges(pair_ids, n_pairs);

    torch::Tensor result = torch::zeros(
        N, 
        torch::TensorOptions().dtype(torch::kFloat32).device(output_grad.device())
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::ScalarType::Half,
        torch::ScalarType::BFloat16,
        input.type(),
        "linear_sd_cuda_backward",
        ([&] {
            linear_sd_backward_kernel<scalar_t><<<grid, PATCH_WA>>>(
                input.data<scalar_t>(),
                output_grad.data<scalar_t>(),
                Ai.data<index_t>(),
                Bi.data<index_t>(),
                std::get<0>(pair_ranges).data<index_t>(),
                std::get<1>(pair_ranges).data<index_t>(),
                Ad,
                Bd,
                h,
                result.data<float>()
            );
        })
    );

    return result.index({inverse_perm});
}


