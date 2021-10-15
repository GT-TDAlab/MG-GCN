#pragma once

#include <cstdlib>
#include <memory>

#include <nccl.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>   
#include <cublas_v2.h>
#include <cassert>
#include <execution>

constexpr auto exec_policy = std::execution::par_unseq;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed on file %s at line %d with error: %s (%d)\n",             \
               __FILE__, __LINE__, cudaGetErrorString(status), status);                  \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed on file %s at line %d with error: %s (%d)\n",         \
               __FILE__, __LINE__, cusparseGetErrorString(status), status);              \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                   \
{                                                                            \
    cublasStatus_t status = (func);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
        printf("CUBLAS API failed on file %s at line %d with error: %s (%d)\n",         \
               __FILE__, __LINE__, cublasGetErrorString(status), status);              \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
}

#define CHECK_NCCL(func) do {                         \
  ncclResult_t r = func;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


template <typename T>
using cuda_ptr = std::shared_ptr<T[]>;

template <typename T>
auto cuda_malloc_managed(std::size_t N) {
	T *p = nullptr;
	if (N) {
        // std::cerr << "Managed allocated " << N * sizeof(T) << std::endl;
        CHECK_CUDA( cudaMallocManaged(reinterpret_cast<void **>(&p), N * sizeof(T)) );
    }
	return cuda_ptr<T>(p, &cudaFree);
}

template <typename T>
auto cuda_malloc(std::size_t N) {
	T *p;
    // std::cerr << "Allocated " << N * sizeof(T) << std::endl;
    CHECK_CUDA( cudaMalloc(reinterpret_cast<void **>(&p), N * sizeof(T)) );
	return cuda_ptr<T>(p, &cudaFree);
}

/** Helper function to return cuda index type parameter **/
template <typename T>
constexpr cusparseIndexType_t get_cusparse_index_type() {
    static_assert(std::is_integral_v<T>, "Integral value is required for cusparse index type.");
    if constexpr (std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t>)
        return CUSPARSE_INDEX_32I;
    if constexpr (std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::int64_t>)
        return CUSPARSE_INDEX_64I;
}

/** Helper function to return cuda data type parameter **/
template <typename T>
constexpr cudaDataType_t get_cuda_data_type() {
    if constexpr (std::is_same_v<T, std::int32_t>)
        return CUDA_R_32I;
    if constexpr (std::is_same_v<T, std::uint32_t>)
        return CUDA_R_32U;
    if constexpr (std::is_same_v<T, std::int64_t>)
        return CUDA_R_64I;
    if constexpr (std::is_same_v<T, std::uint64_t>)
        return CUDA_R_64U;
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    if constexpr (std::is_same_v<T, double>)
        return CUDA_R_64F;
}

template <typename T>
constexpr ncclDataType_t get_nccl_data_type() {
    if constexpr (std::is_same_v<T, std::int32_t>)
        return ncclInt;
    if constexpr (std::is_same_v<T, std::uint32_t>)
        return ncclUint32;
    if constexpr (std::is_same_v<T, std::int64_t>)
        return ncclInt64;
    if constexpr (std::is_same_v<T, std::uint64_t>)
        return ncclUint64;
    if constexpr (std::is_same_v<T, float>)
        return ncclFloat32;
    if constexpr (std::is_same_v<T, double>)
        return ncclDouble;
}