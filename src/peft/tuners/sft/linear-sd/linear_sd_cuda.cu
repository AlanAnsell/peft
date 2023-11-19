#include <cassert>
//#define USE_CUDA_DSA 1

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

const int64_t ELEMENT_RANGES_BLOCKSIZE = 1024;

__global__ void element_ranges_kernel(
    const int64_t* __restrict__ A,
    const int64_t N,
    int64_t* __restrict__ begins,
    int64_t* __restrict__ ends
) {
    __shared__ int64_t values[ELEMENT_RANGES_BLOCKSIZE];
    
    const int64_t idx = threadIdx.x + (blockDim.x - 1) * blockIdx.x;
    int64_t v;
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
            const int64_t prev = values[threadIdx.x - 1];
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
    const int64_t N = sorted_values.size(0);
    torch::Tensor begins = torch::full(
        ub, N,
        torch::TensorOptions().dtype(torch::kInt64).device(sorted_values.device())
    );
    torch::Tensor ends = torch::full(
        ub, N,
        torch::TensorOptions().dtype(torch::kInt64).device(sorted_values.device())
    );
    int64_t num_blocks = (N + ELEMENT_RANGES_BLOCKSIZE - 2) / (ELEMENT_RANGES_BLOCKSIZE - 1);
    element_ranges_kernel<<<dim3(num_blocks), dim3(ELEMENT_RANGES_BLOCKSIZE)>>>(
        sorted_values.data<int64_t>(),
        N,
        begins.data<int64_t>(),
        ends.data<int64_t>()
    );
    return {begins, ends};
}

const int64_t PATCH_WA = 512;
const int64_t PATCH_WB = 128;
const int64_t PATCH_H = 16;

template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int64_t* __restrict__ Ai,
    const int64_t* __restrict__ Bi,
    const int64_t* __restrict__ pair_begins,
    const int64_t* __restrict__ pair_ends,
    const int64_t Ad,
    const int64_t Bd,
    const int64_t h,
    float* __restrict__ outputs
) {
    __shared__ float cached_A[PATCH_H * PATCH_WA];
    __shared__ float cached_B[PATCH_H * PATCH_WB];

    const int64_t Ax = threadIdx.x + PATCH_WA * blockIdx.x;
    const int64_t ybegin = blockIdx.y * PATCH_H;
    const int64_t yend = std::min(ybegin + PATCH_H, h);
    const int64_t yrange = yend - ybegin;

    if (Ax < Ad) {
        for (int64_t i = 0; i < yrange; i++)
            cached_A[threadIdx.x + i * PATCH_WA] = A[Ax + (i + ybegin) * Ad];
    }

    for (int64_t Bx = 0; Bx < Bd; Bx += PATCH_WB) {
        __syncthreads();
        for (int64_t i = threadIdx.x; i < PATCH_H * PATCH_WB; i += PATCH_WA) {
            const int64_t Bcol = Bx + i % PATCH_WB;
            const int64_t Brow = ybegin + i / PATCH_WB;
            if (Bcol < Bd && Brow < h)
                cached_B[i] = B[Bcol + Brow * Bd];
        }
        __syncthreads();

        const int64_t pair_id = blockIdx.x + gridDim.x * (Bx / PATCH_WB);
        //assert(pair_id < N_pairs);
        const int64_t pair_begin = pair_begins[pair_id];
        const int64_t pair_end = pair_ends[pair_id];
        for (int64_t k = pair_begin + threadIdx.x; k < pair_end; k += PATCH_WA) {
            //assert(Ai[k] / PATCH_WA == blockIdx.x);
            //assert(k < N_indices);
            const int64_t Api = Ai[k] - PATCH_WA * blockIdx.x;
            //assert(Api >= 0);
            //assert(Api < PATCH_WA);
            const int64_t Bpi = Bi[k] - Bx;
            //assert(Bpi >= 0);
            //assert(Bpi < PATCH_WB);
            float sum = 0.0;
            for (int64_t i = 0; i < yrange; i++)
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
    const int64_t N = di.size(0);
    //torch::Tensor output_indices = di.floor_divide(input.size(0)); //di / input.size(1);
    //assert(output_indices.dtype() == torch::kInt64);
    //torch::Tensor input_indices = di - input.size(0) * output_indices;
    const int64_t Ad = input.size(1);
    const int64_t Bd = output_grad.size(1);
    const int64_t h = input.size(0);
    const int64_t An = (Ad + PATCH_WA - 1) / PATCH_WA;
    const int64_t Bn = (Bd + PATCH_WB - 1) / PATCH_WB;

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
    torch::Tensor pair_perm = torch::argsort(pair_ids, true);
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
    pair_ids = pair_ids.index({pair_perm});
    //assert(pair_ids.layout() == torch::kStrided && pair_ids.stride(0) == 1);
    //assert(torch::all(
    //    pair_ids.index({torch::indexing::Slice(0, -1)}) <=
    //    pair_ids.index({torch::indexing::Slice(1, torch::indexing::None)})
    //).item<bool>());
    Ai = Ai.index({pair_perm});
    Bi = Bi.index({pair_perm});
    
    const int64_t n_pairs = An * Bn;
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
                Ai.data<int64_t>(),
                Bi.data<int64_t>(),
                std::get<0>(pair_ranges).data<int64_t>(),
                std::get<1>(pair_ranges).data<int64_t>(),
                Ad,
                Bd,
                h,
                result.data<float>()
            );
        })
    );

    return result.index({inverse_perm});
}


