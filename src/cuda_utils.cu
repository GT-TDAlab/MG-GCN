#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <cstdint>
#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <cmath>

/** Only device code here **/
template <typename r_t>
__global__ void relu_forward_kernel(r_t *in, r_t *out, const std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x)
        out[i] = max(in[i], (r_t)0);
}

template <typename r_t>
__global__ void relu_backward_kernel(r_t *in, r_t *G_in, r_t *G_out, const std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x)
        G_out[i] = in[i] > 0 ? G_in[i] : 0;
}

template <typename r_t>
__global__ void leaky_relu_forward_kernel(r_t *in, r_t *out, const std::size_t size, const r_t alpha) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x)
        out[i] = max(in[i], alpha * in[i]);
}

template <typename r_t>
__global__ void leaky_relu_backward_kernel(r_t *in, r_t *G_in, r_t *G_out, const std::size_t size, const r_t alpha) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x)
        G_out[i] = in[i] > 0 ? G_in[i] : alpha * G_in[i];
}

template <typename r_t>
__global__ void broadcast_rows_kernel(r_t *row, r_t *mat, const std::size_t size, const std::size_t m, const bool discard) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = i % m;
    std::size_t j_u = blockDim.x * gridDim.x % m;
    if (discard)
        for (; i < size; i += blockDim.x * gridDim.x, j = j + j_u >= m ? j + j_u - m : j + j_u)
            mat[i] = row[j];
    else
        for (; i < size; i += blockDim.x * gridDim.x, j = j + j_u >= m ? j + j_u - m : j + j_u)
            mat[i] += row[j];
}

template <typename r_t>
__global__ void exp_kernel(r_t *in, r_t *out, const std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x) {
        if constexpr (std::is_same_v<r_t, float>)
            out[i] = expf(in[i]);
        if constexpr (std::is_same_v<r_t, double>)
            out[i] = exp(in[i]);
    }
}

template <typename r_t>
__global__ void log_kernel(r_t *in, r_t *out, const std::size_t size, r_t lowest) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size; i += blockDim.x * gridDim.x) {
        if constexpr (std::is_same_v<r_t, float>)
            out[i] = max(logf(in[i]), lowest);
        if constexpr (std::is_same_v<r_t, double>)
            out[i] = max(log(in[i]), lowest);
    }
}

template <typename r_t>
__global__ void scale_rows_kernel(r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        mat[i] /= scalar[i / m];
}

// B = alpha * A + beta * B
template <typename r_t>
__global__ void axpby_kernel(r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        B[i] = alpha * A[i] + beta * B[i];
}

// B = alpha * A * A + beta * B
template <typename r_t>
__global__ void aaxpby_kernel(r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        B[i] = alpha * A[i] * A[i] + beta * B[i];
}

template <typename r_t>
__global__ void max_rows_kernel(r_t *mat, r_t *maxs, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x) {
        r_t max = -INFINITY;
        for (std::size_t i = 0; i < m; i++)
            max = max > mat[row * m + i] ? max : mat[row * m + i];
        maxs[row] = max;
    }
}

template <typename r_t, typename x_t>
__global__ void max_row_indices_equal_kernel(r_t *mat, r_t *max_values, x_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x) {
        r_t mx = max_values[row];
        std::size_t index = 0;
        for (std::size_t i = 0; i < m; i++)
            if (mx <= mat[row * m + i])
                index = i + rank * m;
        maxs[row] = index;
    }
}

template <typename r_t, typename x_t>
__global__ void max_row_indices_kernel(r_t *mat, x_t *maxs, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x) {
        r_t max = -INFINITY;
        std::size_t index = 0;
        for (std::size_t i = 0; i < m; i++) {
            if (max < mat[row * m + i]) {
                max = mat[row * m + i];
                index = i;
            }
        }
        maxs[row] = index;
    }
}

template <typename r_t, typename x_t>
__global__ void index_rows_kernel(r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x)
        values[row] = mat[row * m + indices[row]];
}

template <typename r_t, typename x_t>
__global__ void index_log_rows_kernel(r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x)
        if constexpr (std::is_same_v<r_t, float>)
            values[row] = logf(mat[row * m + indices[row]]);
        if constexpr (std::is_same_v<r_t, double>)
            values[row] = log(mat[row * m + indices[row]]);
}

template <typename r_t, typename x_t>
__global__ void index_rows_kernel(r_t *mat, x_t *indices, r_t *values, const std::size_t rank, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x)
        values[row] = indices[row] / m == rank ? mat[(row - indices[row] / m) * m + indices[row]] : 0;
}

template <typename r_t, typename x_t>
__global__ void add_indexed_rows_kernel(r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x)
        mat[row * m + indices[row]] += alpha;
}

template <typename r_t, typename x_t>
__global__ void add_indexed_rows_scale_all_kernel(r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m, const r_t scalar) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        mat[i] = (i % m == indices[i / m] ? mat[i] + alpha : mat[i]) * scalar;
}

template <typename r_t, typename x_t>
__global__ void add_indexed_rows_kernel(r_t *mat, x_t *indices, const r_t alpha, const std::size_t rank, const std::size_t size, const std::size_t m) {
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row * m < size; row += blockDim.x * gridDim.x)
        if (indices[row] / m == rank)
            mat[(row - indices[row] / m) * m + indices[row]] += alpha;
}

template <typename r_t, typename x_t>
__global__ void is_equal_kernel(x_t *mat1, x_t *mat2, r_t *out, const std::size_t size) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        out[i] = (r_t)(mat1[i] == mat2[i]);
}

template <typename r_t>
__global__ void subtract_rows_kernel(r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        mat[i] -= scalar[i / m];
}

template <typename r_t>
__global__ void subtract_rows_exp_kernel(r_t *mat, r_t *scalar, r_t *out, const std::size_t size, const std::size_t m) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if constexpr (std::is_same_v<r_t, float>)
            out[i] = expf(mat[i] - scalar[i / m]);
        if constexpr (std::is_same_v<r_t, double>)
            out[i] = exp(mat[i] - scalar[i / m]);
    }
}

template <typename r_t>
__global__ void subtract_kernel(r_t *mat, r_t scalar, const std::size_t size) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        mat[i] -= scalar;
}

template <typename r_t>
__global__ void adam_final_kernel(r_t *param, r_t *m, r_t *v, const r_t step_size, const r_t c2, const r_t eps, const std::size_t size) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        r_t denom;
        if constexpr (std::is_same_v<r_t, float>)
            denom = sqrtf(v[i] / c2) + eps;
        if constexpr (std::is_same_v<r_t, double>)
            denom = sqrt(v[i] / c2) + eps;
        param[i] -= step_size * m[i] / denom;
    }
}

template <typename r_t>
__global__ void transpose_helper_encoder_kernel(r_t *in, r_t *out, const std::size_t size, const std::size_t m, const std::size_t num_procs) {
    const auto mm = m / num_procs;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        auto row = i / m, col = i % m;
        out[(col / mm) * size / num_procs + row * mm + col % mm] = in[i];
    }
}

template <typename r_t>
void leaky_relu_forward(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size, r_t alpha) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    leaky_relu_forward_kernel<r_t><<<blocks, threads, 0, stream>>>(in, out, size, alpha);
}

template <typename r_t>
void leaky_relu_backward(cudaStream_t stream, r_t *in, r_t *G_in, r_t *G_out, const std::size_t size, r_t alpha) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    leaky_relu_backward_kernel<r_t><<<blocks, threads, 0, stream>>>(in, G_in, G_out, size, alpha);
}

template <typename r_t>
void broadcast_rows(cudaStream_t stream, r_t *row, r_t *mat, const std::size_t size, const std::size_t m, const bool discard) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    broadcast_rows_kernel<r_t><<<blocks, threads, 0, stream>>>(row, mat, size, m, discard);
}

template <typename r_t>
void exp(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    exp_kernel<r_t><<<blocks, threads, 0, stream>>>(in, out, size);
}

template <typename r_t>
void log(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    log_kernel<r_t><<<blocks, threads, 0, stream>>>(in, out, size, std::numeric_limits<r_t>::lowest());
}

template <typename r_t>
void scale_rows(cudaStream_t stream, r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    scale_rows_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, scalar, size, m);
}

template <typename r_t>
void max_rows(cudaStream_t stream, r_t *mat, r_t *maxs, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    max_rows_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, maxs, size, m);
}

template <typename r_t, typename x_t>
void max_row_indices_equal(cudaStream_t stream, r_t *mat, r_t *max_values, x_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    max_row_indices_equal_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, max_values, maxs, rank, size, m);    
}

template <typename r_t, typename x_t>
void max_row_indices(cudaStream_t stream, r_t *mat, x_t *maxs, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    max_row_indices_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, maxs, size, m);
}

template <typename r_t, typename x_t>
void index_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    index_rows_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, values, size, m);
}

template <typename r_t, typename x_t>
void index_log_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    index_log_rows_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, values, size, m);
}

template <typename r_t, typename x_t>
void index_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t rank, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    index_rows_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, values, rank, size, m);
}

template <typename r_t, typename x_t>
void add_indexed_rows(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;

    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    add_indexed_rows_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, alpha, size, m);
}

template <typename r_t, typename x_t>
void add_indexed_rows_scale_all(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m, const r_t scalar) {
    const std::size_t threads = 1024;

    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    add_indexed_rows_scale_all_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, alpha, size, m, scalar);
}

template <typename r_t, typename x_t>
void add_indexed_rows(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t rank,const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    add_indexed_rows_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat, indices, alpha, rank, size, m);
}

template <typename r_t, typename x_t>
void is_equal(cudaStream_t stream, x_t *mat1, x_t *mat2, r_t *out, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    is_equal_kernel<r_t, x_t><<<blocks, threads, 0, stream>>>(mat1, mat2, out, size); 
}

template <typename r_t>
void subtract_rows(cudaStream_t stream, r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    subtract_rows_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, scalar, size, m);
}

template <typename r_t>
void subtract_rows_exp(cudaStream_t stream, r_t *mat, r_t *scalar, r_t *out, const std::size_t size, const std::size_t m) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    subtract_rows_exp_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, scalar, out, size, m);
}

template <typename r_t>
void subtract(cudaStream_t stream, r_t *mat, r_t scalar, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    subtract_kernel<r_t><<<blocks, threads, 0, stream>>>(mat, scalar, size);
}

template <typename r_t>
void axpby(cudaStream_t stream, r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    axpby_kernel<r_t><<<blocks, threads, 0, stream>>>(A, B, alpha, beta, size);
}

template <typename r_t>
void aaxpby(cudaStream_t stream, r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    aaxpby_kernel<r_t><<<blocks, threads, 0, stream>>>(A, B, alpha, beta, size);
}

template <typename r_t>
void adam_final(cudaStream_t stream, r_t *param, r_t *m, r_t *v, const r_t lr, const r_t c1, const r_t c2, const r_t eps, const std::size_t size) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    adam_final_kernel<r_t><<<blocks, threads, 0, stream>>>(param, m, v, lr / c1, c2, eps, size);
}

template <typename r_t>
void transpose_helper_encoder(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size, const std::size_t m, const std::size_t num_procs) {
    const std::size_t threads = 1024;
    const std::size_t blocks = std::min((size + threads - 1) / threads, (std::size_t)1280);
    transpose_helper_encoder_kernel<r_t><<<blocks, threads, 0, stream>>>(in, out, size, m, num_procs);    
}

template void leaky_relu_forward<float>(cudaStream_t stream, float *in, float *out, const std::size_t size, float alpha);
template void leaky_relu_forward<double>(cudaStream_t stream, double *in, double *out, const std::size_t size, double alpha);
template void leaky_relu_backward<float>(cudaStream_t stream, float *in, float *G_in, float *G_out, const std::size_t size, float alpha);
template void leaky_relu_backward<double>(cudaStream_t stream, double *in, double *G_in, double *G_out, const std::size_t size, double alpha);
template void broadcast_rows<float>(cudaStream_t stream, float *row, float *mat, const std::size_t size, const std::size_t m, const bool discard);
template void broadcast_rows<double>(cudaStream_t stream, double *row, double *mat, const std::size_t size, const std::size_t m, const bool discard);
template void exp<float>(cudaStream_t stream, float *in, float *out, const std::size_t size);
template void exp<double>(cudaStream_t stream, double *in, double *out, const std::size_t size);
template void log<float>(cudaStream_t stream, float *in, float *out, const std::size_t size);
template void log<double>(cudaStream_t stream, double *in, double *out, const std::size_t size);
template void scale_rows<float>(cudaStream_t stream, float *mat, float *scalar, const std::size_t size, const std::size_t m);
template void scale_rows<double>(cudaStream_t stream, double *mat, double *scalar, const std::size_t size, const std::size_t m);
template void max_rows<float>(cudaStream_t stream, float *mat, float *maxs, const std::size_t size, const std::size_t m);
template void max_rows<double>(cudaStream_t stream, double *mat, double *maxs, const std::size_t size, const std::size_t m);
template void max_row_indices_equal<float, std::int32_t>(cudaStream_t stream, float *mat, float *max_values, std::int32_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m);
template void max_row_indices_equal<float, std::size_t>(cudaStream_t stream, float *mat, float *max_values, std::size_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m);
template void max_row_indices_equal<double, std::int32_t>(cudaStream_t stream, double *mat, double *max_values, std::int32_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m);
template void max_row_indices_equal<double, std::size_t>(cudaStream_t stream, double *mat, double *max_values, std::size_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m);
template void max_row_indices<float, std::size_t>(cudaStream_t stream, float *mat, std::size_t *maxs, const std::size_t size, const std::size_t m);
template void max_row_indices<double, std::size_t>(cudaStream_t stream, double *mat, std::size_t *maxs, const std::size_t size, const std::size_t m);
template void max_row_indices<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *maxs, const std::size_t size, const std::size_t m);
template void max_row_indices<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *maxs, const std::size_t size, const std::size_t m);
template void subtract_rows<float>(cudaStream_t stream, float *mat, float *scalar, const std::size_t size, const std::size_t m);
template void subtract_rows<double>(cudaStream_t stream, double *mat, double *scalar, const std::size_t size, const std::size_t m);
template void subtract_rows_exp<float>(cudaStream_t stream, float *mat, float *scalar, float *out, const std::size_t size, const std::size_t m);
template void subtract_rows_exp<double>(cudaStream_t stream, double *mat, double *scalar, double *out, const std::size_t size, const std::size_t m);
template void subtract<std::uint32_t>(cudaStream_t stream, std::uint32_t *mat, std::uint32_t scalar, const std::size_t size);
template void subtract<std::uint64_t>(cudaStream_t stream, std::uint64_t *mat, std::uint64_t scalar, const std::size_t size);
template void axpby<float>(cudaStream_t stream, float *A, float *B, const float alpha, const float beta, const std::size_t size);
template void axpby<double>(cudaStream_t stream, double *A, double *B, const double alpha, const double beta, const std::size_t size);
template void aaxpby<float>(cudaStream_t stream, float *A, float *B, const float alpha, const float beta, const std::size_t size);
template void aaxpby<double>(cudaStream_t stream, double *A, double *B, const double alpha, const double beta, const std::size_t size);
template void adam_final<float>(cudaStream_t stream, float *param, float *m, float *v, const float lr, const float c1, const float c2, const float eps, const std::size_t size);
template void adam_final<double>(cudaStream_t stream, double *param, double *m, double *v, const double lr, const double c1, const double c2, const double eps, const std::size_t size);
template void index_rows<float, std::size_t>(cudaStream_t stream, float *mat, std::size_t *indices, float *values, const std::size_t size, const std::size_t m);
template void index_rows<double, std::size_t>(cudaStream_t stream, double *mat, std::size_t *indices, double *values, const std::size_t size, const std::size_t m);
template void add_indexed_rows<float, std::size_t>(cudaStream_t stream, float *mat, std::size_t *indices, const float alpha, const std::size_t size, const std::size_t m);
template void add_indexed_rows<double, std::size_t>(cudaStream_t stream, double *mat, std::size_t *indices, const double alpha, const std::size_t size, const std::size_t m);
template void add_indexed_rows<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, const float alpha, const std::size_t size, const std::size_t m);
template void add_indexed_rows<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, const double alpha, const std::size_t size, const std::size_t m);
template void add_indexed_rows_scale_all<float, std::size_t>(cudaStream_t stream, float *mat, std::size_t *indices, const float alpha, const std::size_t size, const std::size_t m, const float scalar);
template void add_indexed_rows_scale_all<double, std::size_t>(cudaStream_t stream, double *mat, std::size_t *indices, const double alpha, const std::size_t size, const std::size_t m, const double scalar);
template void add_indexed_rows_scale_all<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, const float alpha, const std::size_t size, const std::size_t m, const float scalar);
template void add_indexed_rows_scale_all<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, const double alpha, const std::size_t size, const std::size_t m, const double scalar);
template void add_indexed_rows<float, std::size_t>(cudaStream_t stream, float *mat, std::size_t *indices, const float alpha, const std::size_t rank, const std::size_t size, const std::size_t m);
template void add_indexed_rows<double, std::size_t>(cudaStream_t stream, double *mat, std::size_t *indices, const double alpha, const std::size_t rank, const std::size_t size, const std::size_t m);
template void add_indexed_rows<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, const float alpha, const std::size_t rank, const std::size_t size, const std::size_t m);
template void add_indexed_rows<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, const double alpha, const std::size_t rank, const std::size_t size, const std::size_t m);
template void is_equal<float, std::size_t>(cudaStream_t stream, std::size_t *mat1, std::size_t *mat2, float *out, const std::size_t size);
template void is_equal<double, std::size_t>(cudaStream_t stream, std::size_t *mat1, std::size_t *mat2, double *out, const std::size_t size);
template void index_rows<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, float *values, const std::size_t size, const std::size_t m);
template void index_rows<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, double *values, const std::size_t size, const std::size_t m);
template void index_log_rows<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, float *values, const std::size_t size, const std::size_t m);
template void index_log_rows<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, double *values, const std::size_t size, const std::size_t m);
template void index_rows<float, std::int32_t>(cudaStream_t stream, float *mat, std::int32_t *indices, float *values, const std::size_t size, const std::size_t m, const std::size_t rank);
template void index_rows<double, std::int32_t>(cudaStream_t stream, double *mat, std::int32_t *indices, double *values, const std::size_t size, const std::size_t m, const std::size_t rank);
template void is_equal<float, std::int32_t>(cudaStream_t stream, std::int32_t *mat1, std::int32_t *mat2, float *out, const std::size_t size);
template void is_equal<double, std::int32_t>(cudaStream_t stream, std::int32_t *mat1, std::int32_t *mat2,  double *out, const std::size_t size);
template void transpose_helper_encoder<float>(cudaStream_t stream, float *in, float *out, std::size_t size, const std::size_t m, const std::size_t num_procs);
template void transpose_helper_encoder<double>(cudaStream_t stream, double *in, double *out, std::size_t size, const std::size_t m, const std::size_t num_procs);

#endif
