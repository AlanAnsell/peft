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

__global__ void allocate_block_ranges(
    const index_t* __restrict__ row_begins_per_pair,
    const index_t* __restrict__ row_ends_per_pair,
    const index_t* __restrict__ block_ends_per_row,
    const index_t N,
    const index_t step,
    index_t* __restrict__ idx1,
    index_t* __restrict__ idx2_begin,
    index_t* __restrict__ idx2_end
) {
    const index_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        const index_t block_begin = n == 0 ? 0 : block_ends_per_row[n - 1];
        const index_t block_end = block_ends_per_row[n];
        index_t current_pair = row_begins_per_pair[n];
        const index_t last_pair = row_ends_per_pair[n];
        for (index_t i = block_begin; i < block_end; i++) {
            idx1[i] = n;
            idx2_begin[i] = current_pair;
            current_pair = min(current_pair + step, last_pair);
            idx2_end[i] = current_pair;
        }
    }
}


#define DTYPE_PATCH_WA(dtype_bytes) 512
#define DTYPE_PATCH_WB(dtype_bytes) 384
#define DTYPE_PATCH_H(dtype_bytes) 8

const index_t RESULTS_PER_BLOCK = 8;
const index_t THREADS_PER_BLOCK = 64;
const index_t CACHE_SIZE_PER_THREAD = 2;


template <typename scalar_t>
__global__ void linear_sd_backward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const index_t* __restrict__ Ai,
    const index_t* __restrict__ Bi,
    const index_t* __restrict__ Bbegins,
    const index_t* __restrict__ Bends,
    const index_t h,
    float* __restrict__ outputs
) {
    __shared__ float reduced_sums[RESULTS_PER_BLOCK][THREADS_PER_BLOCK + 1];

    const index_t Arow = Ai[blockIdx.x] * h;
    const index_t Bbegin = Bbegins[blockIdx.x];
    const index_t Bend = Bends[blockIdx.x];
    const index_t Brange = Bend - Bbegin;
    //assert(Brange <= RESULTS_PER_BLOCK);
    
    //scalar_t Acache[CACHE_SIZE_PER_THREAD];
    float partial_sums[RESULTS_PER_BLOCK] = {0.0};
    index_t Brows[RESULTS_PER_BLOCK];

    for (index_t i = 0; i < RESULTS_PER_BLOCK; i++) {
        if (i >= Brange)
            break;
        Brows[i] = Bi[i + Bbegin] * h;
    }
    
    for (index_t first_col = threadIdx.x; first_col < h; first_col += 2 * THREADS_PER_BLOCK) {
        //index_t col = first_col;
        //for (index_t i = 0; i < CACHE_SIZE_PER_THREAD; i++) {
        //    if (col >= h)
        //        break;
        //    Acache[i] = A[col + Arow];
        //    col += THREADS_PER_BLOCK;
        //}

        const bool second_valid = first_col + THREADS_PER_BLOCK < h;
        const scalar_t Aval1 = A[first_col + Arow];
        const scalar_t Aval2 = second_valid * A[second_valid * THREADS_PER_BLOCK + first_col + Arow];
        for (index_t i = 0; i < RESULTS_PER_BLOCK; i++) {
            if (i >= Brange)
                break;
            const index_t Brow = Brows[i];
            //col = first_col;
            //for (index_t j = 0; j < CACHE_SIZE_PER_THREAD; j++) {
            //    if (col >= h)
            //        break;
            //    partial_sums[i] += Acache[j] * B[col + Brow];
            //    col += THREADS_PER_BLOCK;
            //}
            partial_sums[i] += Aval1 * B[first_col + Brow] + Aval2 * B[second_valid * THREADS_PER_BLOCK + first_col + Brow];
        }
    }

    for (index_t i = 0; i < RESULTS_PER_BLOCK; i++)
        reduced_sums[i][threadIdx.x] = partial_sums[i];
    __syncthreads();

    for (index_t i = threadIdx.x; i < Brange; i += THREADS_PER_BLOCK) {
        float sum = 0.0;
        for (index_t j = 0; j < THREADS_PER_BLOCK; j++)
            sum += reduced_sums[i][j];
        outputs[i + Bbegin] = sum;
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

    auto Bi_ranges_per_pair = element_ranges(Bi, Bd);
    torch::Tensor Bi_begins_per_pair = std::get<0>(Bi_ranges_per_pair);
    torch::Tensor Bi_ends_per_pair = std::get<1>(Bi_ranges_per_pair);
    torch::Tensor Bi_row_counts = Bi_ends_per_pair - Bi_begins_per_pair;
    torch::Tensor Bi_block_counts = (Bi_row_counts + (RESULTS_PER_BLOCK - 1)).floor_divide(RESULTS_PER_BLOCK);
    torch::Tensor Bi_block_ends = torch::cumsum(Bi_block_counts, 0, torch::kInt32);
    const index_t n_blocks = Bi_block_counts.sum().item<index_t>();
    //std::cerr << n_blocks << " blocks" << std::endl;

    torch::Tensor Brows = torch::empty(
        {n_blocks},
        torch::TensorOptions().dtype(torch::kInt32).device(output_grad.device())
    );
    torch::Tensor Abegins = torch::empty(
        {n_blocks},
        torch::TensorOptions().dtype(torch::kInt32).device(output_grad.device())
    );
    torch::Tensor Aends = torch::empty(
        {n_blocks},
        torch::TensorOptions().dtype(torch::kInt32).device(output_grad.device())
    );
    allocate_block_ranges<<<(Bd + 31) / 32, 32>>>(
        Bi_begins_per_pair.data<index_t>(),
        Bi_ends_per_pair.data<index_t>(),
        Bi_block_ends.data<index_t>(),
        Bd,
        RESULTS_PER_BLOCK,
        Brows.data<index_t>(),
        Abegins.data<index_t>(),
        Aends.data<index_t>()
    );
    //for (index_t i = 0; i < n_blocks; i++) {
    //    std::cerr << Brows.index({i}).item<index_t>() << ", "
    //              << Abegins.index({i}).item<index_t>() << ", "
    //              << Aends.index({i}).item<index_t>() << std::endl;
    //}


    //torch::Tensor Bbegins = 
    //const index_t dtype_bytes = torch::elementSize(torch::typeMetaToScalarType(input.dtype()));
    //const index_t PATCH_WA = DTYPE_PATCH_WA(dtype_bytes);
    //const index_t PATCH_WB = DTYPE_PATCH_WB(dtype_bytes);
    //const index_t PATCH_H = DTYPE_PATCH_H(dtype_bytes);

    //const index_t An = (Ad + PATCH_WA - 1) / PATCH_WA;
    //const index_t Bn = (Bd + PATCH_WB - 1) / PATCH_WB;

    const dim3 grid(n_blocks);

    torch::Tensor result = torch::empty(
        N, 
        torch::TensorOptions().dtype(torch::kFloat32).device(output_grad.device())
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::ScalarType::Half,
        torch::ScalarType::BFloat16,
        input.type(),
        "linear_sd_cuda_backward",
        ([&] {
            linear_sd_backward_kernel<scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                output_grad.data<scalar_t>(),
                input.data<scalar_t>(),
                Brows.data<index_t>(),
                Ai.data<index_t>(),
                Abegins.data<index_t>(),
                Aends.data<index_t>(),
                h,
                result.data<float>()
            );
        })
    );

    return result; //.index({inverse_perm});
}


