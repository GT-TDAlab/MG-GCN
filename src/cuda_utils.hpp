#pragma once

#include <cstdlib>
#include <memory>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>   
#include <cublas_v2.h>

#include "matrix.hpp"
#include "dist_matrix.hpp"
#include "mg_gcn.hpp"

/**
 * SpMM performs C = alpha * A * B + beta * C
 * 
 * @param A csr matrix
 * @param B dense matrix
 * @param C dense matrix
 * @param alpha scalar multiplier for A
 * @param beta scalar multiplier for B
 * @param alg SpMM algorithm. For details https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm
 * @tested
 * 
 */ 
template <typename x_t, typename v_t, typename r_t>
void matmul(const context ctx, const csr_matrix<x_t, v_t, r_t> A, const dn_matrix<r_t> B, const dn_matrix<r_t> C, const cuda_ptr<char> ext_buffer, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(A.m() == B.n() && B.m() == C.m());
    ctx.set();
    CHECK_CUSPARSE( cusparseSpMM(ctx.cusparse_handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.get_mat(), B.get_mat(), &beta, C.get_mat(), get_cuda_data_type<r_t>(), alg, ext_buffer.get()) );
}

template <typename x_t, typename v_t, typename r_t>
void matmul(const dist_context ctx, dist_csr_matrix<x_t, v_t, r_t> A, dist_dn_matrix<r_t> B, dist_dn_matrix<r_t> C, std::vector<cuda_ptr<char>> ext_buffer, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(ctx.size() == A.size() && A.size() == B.size() && B.size() == C.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < ctx.size(); i++) {
        A.bcast(i);
        auto Cs = C.part2d(offset, A[0].n());
        for (std::size_t j = 0; j < ctx.size(); j++)
            matmul(ctx[j], A[j], B[j], Cs[j], ext_buffer[j], alpha, beta, alg);
        offset += A[0].n();
    }
}

template <typename x_t, typename v_t, typename r_t>
void matmul(const dist_context ctx, dist_row_csr_matrix<x_t, v_t, r_t> A, dist_row_dn_matrix<r_t> B, dist_row_dn_matrix<r_t> C, std::vector<cuda_ptr<char>> ext_buffer, dist_row_dn_matrix<r_t> B_bcast, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(ctx.size() == A.size() && A.size() == B.size() && B.size() == C.size());
    for (std::size_t i = 0; i < ctx.size(); i++) {
        B.bcast(ctx, i, B_bcast);
        for (std::size_t j = 0; j < ctx.size(); j++)
            matmul(ctx[j], A[{j, i}], B_bcast[j], C[j], ext_buffer[j], alpha, i == 0 ? beta : (r_t)1, alg);
    }
}

template <typename x_t, typename v_t, typename r_t>
void matmul(dist_context ctx, dist_row_csr_matrix<x_t, v_t, r_t> A, dist_row_dn_matrix<r_t> B, dist_row_dn_matrix<r_t> C, std::vector<cuda_ptr<char>> ext_buffer, std::vector<dist_row_dn_matrix<r_t>> B_bcast, const r_t alpha, const r_t beta, const std::string name = "", const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(ctx.size() == A.size() && A.size() == B.size() && B.size() == C.size());
    
    ctx.record(name + "0_matmul-spmm", 0);
    ctx.wait(name + "0_matmul-spmm", 1);
    // ctx.register_walltime(name + "0_matmul-spmm", 1);

    for (std::size_t i = 0; i < ctx.size(); i++) {
        if (i > 0)
            ctx.wait(name + std::to_string(i - 1) + "_matmul-spmm", ctx.bcast_stream_id());

        ctx.record(name + std::to_string(i) + "_matmul-bcast-start", ctx.bcast_stream_id());
        
        B.bcast(ctx, i, B_bcast[i % 2], ctx.bcast_stream_id());

        ctx.record(name + std::to_string(i) + "_matmul-bcast-finish", ctx.bcast_stream_id());

        ctx.register_timer(name + std::to_string(i) + "_matmul-bcast-start", "training-start", name + std::to_string(i) + "_matmul-bcast-start");
        ctx.register_timer(name + std::to_string(i) + "_matmul-bcast-finish", "training-start", name + std::to_string(i) + "_matmul-bcast-finish");

        ctx.wait(name + std::to_string(i) + "_matmul-bcast-finish", 0);
        
        ctx.record(name + std::to_string(i + 1) + "_matmul-spmm-start", 0);

        for (std::size_t j = 0; j < ctx.size(); j++)
            matmul(ctx[j], A[{j, i}], B_bcast[i % 2][j], C[j], ext_buffer[j], alpha, i == 0 ? beta : (r_t)1, alg);
        
        ctx.record(name + std::to_string(i + 1) + "_matmul-spmm", 0);

        ctx.register_timer(name + std::to_string(i) + "_matmul-spmm-start", "training-start", name + std::to_string(i + 1) + "_matmul-spmm-start");
        ctx.register_timer(name + std::to_string(i) + "_matmul-spmm-finish", "training-start", name + std::to_string(i + 1) + "_matmul-spmm");
    }
    
    ctx.register_timer(name + "matmul-spmm", name + "0_matmul-spmm", name + std::to_string(ctx.size()) + "_matmul-spmm");
}

template <typename x_t, typename v_t, typename r_t>
auto get_matmul_buffer(const context ctx, const csr_matrix<x_t, v_t, r_t> A, const dn_matrix<r_t> B, const dn_matrix<r_t> C, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() && B.m() == C.m());
    std::size_t ext_buffer_size = 0;
    ctx.set();
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(ctx.cusparse_handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.get_mat(), B.get_mat(), &beta, C.get_mat(), get_cuda_data_type<r_t>(), alg, &ext_buffer_size) );
    return cuda_malloc<char>(ext_buffer_size);
}

template <typename x_t, typename v_t, typename r_t>
auto get_matmul_buffer(const dist_context ctx, const dist_csr_matrix<x_t, v_t, r_t> A, const dist_dn_matrix<r_t> B, const dist_dn_matrix<r_t> C, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() && B.m() == C.m());
    std::vector<std::size_t> ext_buffer_sizes(ctx.size(), 0);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < ctx.size(); i++) {
        const auto Cs = C.part2d(offset, A(i).n());
        for (std::size_t j = 0; j < ctx.size(); j++) {
            ctx[j].set();
            std::size_t ext_buffer_size;
            CHECK_CUSPARSE( cusparseSpMM_bufferSize(ctx[j].cusparse_handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A(i).get_mat(), B[j].get_mat(), &beta, Cs[j].get_mat(), get_cuda_data_type<r_t>(), alg, &ext_buffer_size) );
            ext_buffer_sizes[j] = std::max(ext_buffer_sizes[j], ext_buffer_size);
        }
        offset += A(i).n();
    }
    std::vector<cuda_ptr<char>> ext_buffers;
    for (std::size_t i = 0; i < ctx.size(); i++) {
        ctx[i].set();
        ext_buffers.emplace_back(cuda_malloc<char>(ext_buffer_sizes[i]));
    }
    return ext_buffers;
}

template <typename x_t, typename v_t, typename r_t>
auto get_matmul_buffer(const dist_context ctx, const dist_row_csr_matrix<x_t, v_t, r_t> A, dist_row_dn_matrix<r_t> B, const dist_row_dn_matrix<r_t> C, const r_t alpha, const r_t beta, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() && B.m() == C.m());
    std::vector<std::size_t> ext_buffer_sizes(ctx.size(), 0);
    for (std::size_t i = 0; i < ctx.size(); i++) {
        for (std::size_t j = 0; j < ctx.size(); j++) {
            ctx[j].set();
            std::size_t ext_buffer_size;
            CHECK_CUSPARSE( cusparseSpMM_bufferSize(ctx[j].cusparse_handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A[{j, i}].get_mat(), B[j].get_mat(), &beta, C[j].get_mat(), get_cuda_data_type<r_t>(), alg, &ext_buffer_size) );
            ext_buffer_sizes[j] = std::max(ext_buffer_sizes[j], ext_buffer_size);
        }
    }
    std::vector<cuda_ptr<char>> ext_buffers;
    for (std::size_t i = 0; i < ctx.size(); i++) {
        ctx[i].set();
        ext_buffers.emplace_back(cuda_malloc<char>(ext_buffer_sizes[i]));
    }
    return ext_buffers;
}

/**
 * GeMM performs C = alpha * A * B + beta * C
 * 
 * @param A dense matrix
 * @param B dense matrix
 * @param alpha scalar multiplier for A
 * @param beta scalar multiplier for B
 * 
 */ 
template <typename r_t>
void matmul(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B, const dn_matrix<r_t> C, const r_t alpha, const r_t beta, const bool A_T = false, const bool B_T = false) {
    auto A_n = A.n(), A_m = A.m(), B_n = B.n(), B_m = B.m();
    if (A_T)
        std::swap(A_n, A_m);
    if (B_T)
        std::swap(B_n, B_m);
    assert(A_m == B_n);
    assert(A_n == C.n() && B_m == C.m());
    ctx.set();
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSgemm(ctx.cublas_handle.get(), B_T ? CUBLAS_OP_T : CUBLAS_OP_N, A_T ? CUBLAS_OP_T : CUBLAS_OP_N, B_m, A_n, A_m, &alpha, B.buffer(), B.m(), A.buffer(), A.m(), &beta, C.buffer(), C.m()) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDgemm(ctx.cublas_handle.get(), B_T ? CUBLAS_OP_T : CUBLAS_OP_N, A_T ? CUBLAS_OP_T : CUBLAS_OP_N, B_m, A_n, A_m, &alpha, B.buffer(), B.m(), A.buffer(), A.m(), &beta, C.buffer(), C.m()) );
}

template <typename r_t>
void matmul(const dist_context ctx, const dist_dn_matrix<r_t> A, const dist_dn_matrix<r_t> B, const dist_dn_matrix<r_t> C, const r_t alpha, const r_t beta, std::vector<dn_matrix<r_t>> tA) {
    assert(A.n() == B.n() && A.m() == C.n() && B.m() == C.m());
    assert(A.m() % ctx.size() == 0);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < ctx.size(); i++) {
        CHECK_NCCL( ncclGroupStart() );
        for (std::size_t j = 0; j < ctx.size(); j++)
            CHECK_NCCL( ncclBroadcast(A[i].buffer(), tA[j].buffer(), A[i].size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
        CHECK_NCCL( ncclGroupEnd() );
        auto Cs = C.part2d(offset, A[i].m());
        for (std::size_t j = 0; j < ctx.size(); j++)
            matmul(ctx[j], tA[j], B[j], Cs[j], alpha, beta, true);
        offset += A[i].m();
    }
}


template <typename r_t>
void matmul(const dist_context ctx, const dist_dn_matrix<r_t> A, const dist_dn_matrix<r_t> B, const dist_dn_matrix<r_t> C, const r_t alpha, const r_t beta, std::vector<dn_matrix<r_t>> tB, std::vector<dn_matrix<r_t>> tC) {
    assert(A.m() == B.n() && A.n() == C.n() && B.m() == C.m());
    assert(B.n() % ctx.size() == 0 && B.m() % ctx.size() == 0);
    ctx.sync();
    for (std::size_t j = 0; j < ctx.size(); j++) {
        {
            // timer t("Scatter", &ctx);
            // CHECK_NCCL( ncclGroupStart() );
            // for (std::size_t i = 0; i < ctx.size(); i++) {
            //     CHECK_NCCL( ncclSend(B[j].buffer() + i * B[j].size() / ctx.size(), B[j].size() / ctx.size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
            //     CHECK_NCCL( ncclRecv(tB[i].buffer(), B[j].size() / ctx.size(), get_nccl_data_type<r_t>(), j, ctx(i), ctx[i].cuda_streams->get()) );
            // }
            // CHECK_NCCL( ncclGroupEnd() );
            for (std::size_t i = 0; i < ctx.size(); i++)
                tB[i] = dn_matrix<r_t>(tB[i].n(), tB[i].m(), cuda_ptr<r_t>(B[j].shared_buffer(), B[j].buffer() + i * B[j].size() / ctx.size()));
        }

        // ctx.sync();

        {
            timer t("matmul", &ctx);
            for (std::size_t i = 0; i < ctx.size(); i++)
                matmul(ctx[i], A[i], tB[i], tC[i], alpha, (r_t)0);
        }

        {
            timer t("Reduce", &ctx);
            CHECK_NCCL( ncclGroupStart() );
            for (std::size_t i = 0; i < ctx.size(); i++)
                CHECK_NCCL( ncclReduce(tC[i].buffer(), tC[i].buffer(), tC[i].size(), get_nccl_data_type<r_t>(), ncclSum, j, ctx(i), ctx[i].cuda_streams->get()) );
            CHECK_NCCL( ncclGroupEnd() );
        }

        timer t("axpby", &ctx);
        axpby(ctx[j], tC[j], C[j], (r_t)1, beta);
    }
}

// template <typename r_t>
// void matmul(const dist_context ctx, const dist_dn_matrix<r_t> A, const repl_dn_matrix<r_t> B, const dist_dn_matrix<r_t> C, const r_t alpha, const r_t beta, std::vector<dn_matrix<r_t>> tB, std::vector<dn_matrix<r_t>> tC) {
//     assert(A.m() == B.n() && A.n() == C.n() && B.m() == C.m());
//     assert(B.n() % ctx.size() == 0 && B.m() % ctx.size() == 0);
//     for (std::size_t j = 0; j < ctx.size(); j++) {
//         {
//             timer t("Scatter", &ctx);
//             CHECK_NCCL( ncclGroupStart() );
//             for (std::size_t i = 0; i < ctx.size(); i++) {
//                 CHECK_NCCL( ncclSend(B[j].buffer() + i * B[j].size() / ctx.size(), B[j].size() / ctx.size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
//                 CHECK_NCCL( ncclRecv(tB[i].buffer(), B[j].size() / ctx.size(), get_nccl_data_type<r_t>(), j, ctx(i), ctx[i].cuda_streams->get()) );
//             }
//             CHECK_NCCL( ncclGroupEnd() );
//         }

//         {
//             timer t("matmul", &ctx);
//             for (std::size_t i = 0; i < ctx.size(); i++)
//                 matmul(ctx[i], A[i], tB[i], tC[i], alpha, (r_t)0);
//         }

//         {
//             timer t("Reduce", &ctx);
//             CHECK_NCCL( ncclGroupStart() );
//             for (std::size_t i = 0; i < ctx.size(); i++)
//                 CHECK_NCCL( ncclReduce(tC[i].buffer(), tC[i].buffer(), tC[i].size(), get_nccl_data_type<r_t>(), ncclSum, j, ctx(i), ctx[i].cuda_streams->get()) );
//             CHECK_NCCL( ncclGroupEnd() );
//         }

//         timer t("axpby", &ctx);
//         axpby(ctx[j], tC[j], C[j], (r_t)1, beta);
//     }
// }


template <typename r_t>
void matmul(const dist_context ctx, const dist_dn_matrix<r_t> A, const dist_dn_matrix<r_t> B, const dist_dn_matrix<r_t> C, const r_t alpha, const r_t beta, const bool A_T = false, const bool B_T = false) {
    auto A_n = A.n(), A_m = A.m(), B_n = B.n(), B_m = B.m();
    if (A_T)
        std::swap(A_n, A_m);
    if (B_T)
        std::swap(B_n, B_m);
    assert(A_m == B_n);
    assert(A_n == C.n() && B_m == C.m());
    auto BB = B;
    if (B_T)
        BB = B.transpose(ctx);
    if (A_T) {
        assert(A.m() % ctx.size() == 0);
        std::vector<dn_matrix<r_t>> tA;
        for (std::size_t i = 0; i < ctx.size(); i++) {
            ctx[i].set();
            tA.emplace_back(A[i].shape(), cuda_malloc<r_t>(A[i].size()));
        }
        matmul(ctx, A, BB, C, alpha, beta, tA);
    }
    else {
        auto AA = A;

        assert(BB.n() % ctx.size() == 0 && BB.m() % ctx.size() == 0);

        ctx.sync();

        std::vector<dn_matrix<r_t>> tC, tB;

        for (std::size_t i = 0; i < ctx.size(); i++) {
            ctx[i].set();
            tC.emplace_back(C.n(), C.m() / ctx.size(), cuda_malloc<r_t>(C.n() * C.m() / ctx.size()));
            tB.emplace_back(BB.n() / ctx.size(), BB.m() / ctx.size(), cuda_malloc<r_t>(BB.n() / ctx.size() * BB.m() / ctx.size()));
        }
        matmul(ctx, AA, BB, C, alpha, beta, tB, tC);
    }
}
template <typename r_t>
void matmul(const dist_context ctx, const dist_row_dn_matrix<r_t> A, const dist_row_dn_matrix<r_t> B, const repl_dn_matrix<r_t> C, const r_t alpha, const r_t beta) {
    auto A_n = A.n(), A_m = A.m(), B_n = B.n(), B_m = B.m();
    std::swap(A_n, A_m);
    assert(A_m == B_n);
    assert(A_n == C.n() && B_m == C.m());
    for (std::size_t i = 0; i < ctx.size(); i++)
        matmul(ctx[i], A[i], B[i], C[i], alpha, beta, true);
    C.allreduce(ctx);
}

template <typename r_t>
void matmul(const dist_context ctx, const dist_row_dn_matrix<r_t> A, const repl_dn_matrix<r_t> B, const dist_row_dn_matrix<r_t> C, const r_t alpha, const r_t beta, const bool B_T = false) {
    auto A_n = A.n(), A_m = A.m(), B_n = B.n(), B_m = B.m();
    if (B_T)
        std::swap(B_n, B_m);
    assert(A_m == B_n);
    assert(A_n == C.n() && B_m == C.m());
    for (std::size_t i = 0; i < ctx.size(); i++)
        matmul(ctx[i], A[i], B[i], C[i], alpha, beta, false, B_T);
}

template <typename r_t>
void axpy(const context ctx, const r_t *A, r_t *B, const std::size_t size, const r_t alpha, const int inc_A, const int inc_B) {
    ctx.set();
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSaxpy(ctx.cublas_handle.get(), size, &alpha, A, inc_A, B, inc_B) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDaxpy(ctx.cublas_handle.get(), size, &alpha, A, inc_A, B, inc_B) );
}

// @tested
// B += alpha * A
template <typename r_t>
void axpy(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B, const r_t alpha, const int inc_A = 1, const int inc_B = 1) {
    assert(A.shape() == B.shape());
    ctx.set();
    axpy(ctx, A.buffer(), B.buffer(), A.n() * A.m(), alpha, inc_A, inc_B);
}

template <typename r_t, template<typename T> class dn_t>
void axpy(const dist_context ctx, const dn_t<r_t> A, const dn_t<r_t> B, const r_t alpha, const int inc_A = 1, const int inc_B = 1) {
    assert(A.shape() == B.shape());
    for (std::size_t i = 0; i < ctx.size(); i++)
        axpy(ctx[i], A[i], B[i], alpha, inc_A, inc_B);
}

// @tested
template <typename r_t>
r_t dot(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B) {
    assert(A.shape() == B.shape());
    ctx.set();
    r_t result = 0;
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSdot(ctx.cublas_handle.get(), A.n() * B.m(), A.buffer(), 1, B.buffer(), 1, &result) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDdot(ctx.cublas_handle.get(), A.n() * B.m(), A.buffer(), 1, B.buffer(), 1, &result) );
    return result;
}

template <typename r_t>
void abssum(const context ctx, const dn_matrix<r_t> A, r_t &result) {
    ctx.set();
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSasum(ctx.cublas_handle.get(), A.size(), A.buffer(), 1, &result) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDasum(ctx.cublas_handle.get(), A.size(), A.buffer(), 1, &result) );
}

// @tested
template <typename r_t>
void scale_mat(const context ctx, const dn_matrix<r_t> mat, r_t scalar) {
    ctx.set();
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSscal(ctx.cublas_handle.get(), mat.n() * mat.m(), &scalar, mat.buffer(), 1) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDscal(ctx.cublas_handle.get(), mat.n() * mat.m(), &scalar, mat.buffer(), 1) );
}

template <typename r_t, template<typename T> class dn_t>
void scale_mat(const dist_context ctx, const dn_t<r_t> mat, r_t scalar) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        scale_mat(ctx[i], mat[i], scalar);
}

template <typename r_t>
void geam(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B, const dn_matrix<r_t> C, const r_t alpha, const r_t beta, const bool T1, const bool T2) {
    ctx.set();
    if constexpr (std::is_same_v<r_t, float>)
        CHECK_CUBLAS( cublasSgeam(ctx.cublas_handle.get(), !T1 ? CUBLAS_OP_N : CUBLAS_OP_T, !T2 ? CUBLAS_OP_N : CUBLAS_OP_T, C.m(), C.n(), &alpha, A.buffer(), A.m(), &beta, B.buffer(), B.m(), C.buffer(), C.m()) );
    if constexpr (std::is_same_v<r_t, double>)
        CHECK_CUBLAS( cublasDgeam(ctx.cublas_handle.get(), !T1 ? CUBLAS_OP_N : CUBLAS_OP_T, !T2 ? CUBLAS_OP_N : CUBLAS_OP_T, C.m(), C.n(), &alpha, A.buffer(), A.m(), &beta, B.buffer(), B.m(), C.buffer(), C.m()) );
}

template <typename r_t>
void leaky_relu_forward(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size, r_t alpha);

template <typename r_t>
void leaky_relu_backward(cudaStream_t stream, r_t *in, r_t *G_in, r_t *G_out, const std::size_t size, r_t alpha);

// @tested
template <typename r_t>
void broadcast_rows(cudaStream_t stream, r_t *row, r_t *mat, const std::size_t size, const std::size_t m, const bool discard);

template <typename r_t>
void exp(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size);

template <typename r_t>
void log(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size);

// @tested
template <typename r_t>
void scale_rows(cudaStream_t stream, r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m);

template <typename r_t>
void max_rows(cudaStream_t stream, r_t *mat, r_t *maxs, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void max_row_indices_equal(cudaStream_t stream, r_t *mat, r_t *max_values, x_t *maxs, const std::size_t rank, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void max_row_indices(cudaStream_t stream, r_t *mat, x_t *maxs, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void index_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void index_log_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void index_rows(cudaStream_t stream, r_t *mat, x_t *indices, r_t *values, const std::size_t size, const std::size_t m, const std::size_t rank);

template <typename r_t, typename x_t>
void add_indexed_rows(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void add_indexed_rows_scale_all(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t size, const std::size_t m, const r_t scalar);

template <typename r_t, typename x_t>
void add_indexed_rows(cudaStream_t stream, r_t *mat, x_t *indices, const r_t alpha, const std::size_t rank, const std::size_t size, const std::size_t m);

template <typename r_t, typename x_t>
void is_equal(cudaStream_t stream, x_t *mat1, x_t *mat2, r_t *out, const std::size_t size);

// @tested
template <typename r_t>
void subtract_rows(cudaStream_t stream, r_t *mat, r_t *scalar, const std::size_t size, const std::size_t m);

template <typename r_t>
void subtract_rows_exp(cudaStream_t stream, r_t *mat, r_t *scalar, r_t *out, const std::size_t size, const std::size_t m); 

template <typename r_t>
void subtract(cudaStream_t stream, r_t *mat, r_t scalar, const std::size_t size);

template <typename r_t>
void axpby(cudaStream_t stream, r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size);

template <typename r_t>
void aaxpby(cudaStream_t stream, r_t *A, r_t *B, const r_t alpha, const r_t beta, const std::size_t size);

template <typename r_t>
void adam_final(cudaStream_t stream, r_t *param, r_t *m, r_t *v, const r_t lr, const r_t c1, const r_t c2, const r_t eps, const std::size_t size);

template <typename r_t>
void transpose_helper_encoder(cudaStream_t stream, r_t *in, r_t *out, const std::size_t size, const std::size_t m, std::size_t num_procs);

template <typename r_t>
void leaky_relu_forward(const context ctx, const dn_matrix<r_t> in, const dn_matrix<r_t> out, r_t alpha = 0.01) {
    ctx.set();
    assert(in.n() == out.n() && in.m() == out.m());
    const std::size_t size = in.n() * in.m();
    leaky_relu_forward(ctx.cuda_streams->get(), in.buffer(), out.buffer(), size, alpha);
}

template <typename r_t, template<typename T> class dn_t>
void leaky_relu_forward(const dist_context ctx, const dn_t<r_t> in, const dn_t<r_t> out, r_t alpha = 0.01) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        leaky_relu_forward(ctx[i], in[i], out[i], alpha);
}

template <typename r_t>
void leaky_relu_backward(const context ctx, const dn_matrix<r_t> in, const dn_matrix<r_t> G_in, const dn_matrix<r_t> G_out, r_t alpha = 0.01) {
    ctx.set();
    assert(in.n() == G_in.n() && in.m() == G_in.m());
    assert(in.n() == G_out.n() && in.m() == G_out.m());
    auto size = in.n() * in.m();
    leaky_relu_backward(ctx.cuda_streams->get(), in.buffer(), G_in.buffer(), G_out.buffer(), size, alpha);
}

template <typename r_t, template<typename T> class dn_t>
void leaky_relu_backward(const dist_context ctx, const dn_t<r_t> in, const dn_t<r_t> G_in, const dn_t<r_t> G_out, r_t alpha = 0.01) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        leaky_relu_backward(ctx[i], in[i], G_in[i], G_out[i], alpha);
}

template <typename r_t>
void broadcast_rows(const context ctx, const dn_matrix<r_t> row, const dn_matrix<r_t> mat, const bool discard = true) {
    ctx.set();
    assert(row.m() == mat.m());
    const std::size_t size = mat.n() * mat.m();
    broadcast_rows(ctx.cuda_streams->get(), row.buffer(), mat.buffer(), size, mat.m(), discard);
}

template <typename r_t, template<typename T> class dn1_t, template<typename T> class dn2_t>
void broadcast_rows(const dist_context ctx, const dn1_t<r_t> row, const dn2_t<r_t> mat, const bool discard = true) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        broadcast_rows(ctx[i], row[i], mat[i], discard);
}

template <typename r_t>
void exp(const context ctx, dn_matrix<r_t> in, dn_matrix<r_t> out) {
    ctx.set();
    assert(in.n() == out.n() && in.m() == out.m());
    exp(ctx.cuda_streams->get(), in.buffer(), out.buffer(), in.n() * in.m());
}

template <typename r_t, template<typename T> class dn_t>
void exp(const dist_context ctx, dn_t<r_t> in, dn_t<r_t> out) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        exp(ctx[i], in[i], out[i]);
}

template <typename r_t>
void log(const context ctx, dn_matrix<r_t> in, dn_matrix<r_t> out) {
    ctx.set();
    assert(in.n() == out.n() && in.m() == out.m());
    log(ctx.cuda_streams->get(), in.buffer(), out.buffer(), in.n() * in.m());
}

template <typename r_t, template<typename T> class dn_t>
void log(const dist_context ctx, dn_t<r_t> in, dn_t<r_t> out) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        log(ctx[i], in[i], out[i]);
}

template <typename r_t>
void scale_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<r_t> scalar) {
    ctx.set();
    assert(mat.n() == scalar.n());
    const std::size_t size = mat.n() * mat.m();
    scale_rows(ctx.cuda_streams->get(), mat.buffer(), scalar.buffer(), size, mat.m());
}

template <typename r_t>
void scale_rows(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dist_dn_matrix<r_t> scalar) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        scale_rows(ctx[i], mat[i], scalar[i]);
}

template <typename r_t>
void max_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<r_t> maxs) {
    ctx.set();
    assert(mat.n() == maxs.n() && maxs.m() == 1);
    max_rows(ctx.cuda_streams->get(), mat.buffer(), maxs.buffer(), mat.size(), mat.m()); 
}

template <typename r_t>
void max_rows(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dist_dn_matrix<r_t> maxs) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        max_rows(ctx[i], mat[i], maxs[i]);
    
    CHECK_NCCL( ncclGroupStart() );
    for (std::size_t i = 0; i < ctx.size(); i++)
        CHECK_NCCL( ncclAllReduce(maxs[i].buffer(), maxs[i].buffer(), maxs[i].size(), get_nccl_data_type<r_t>(), ncclMax, ctx(i), ctx[i].cuda_streams->get()) );
    CHECK_NCCL( ncclGroupEnd() );
}

template <typename r_t, typename x_t>
void max_row_indices(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> maxs) {
    ctx.set();
    assert(mat.n() == maxs.n() && maxs.m() == 1);
    max_row_indices(ctx.cuda_streams->get(), mat.buffer(), maxs.buffer(), mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void max_row_indices_equal(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<r_t> max_values, const dn_matrix<x_t> maxs, const std::size_t rank) {
    assert(mat.n() == max_values.n() && max_values.shape() == maxs.shape() && maxs.m() == 1);
    ctx.set();
    max_row_indices_equal(ctx.cuda_streams->get(), mat.buffer(), max_values.buffer(), maxs.buffer(), rank, mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void max_row_indices_equal(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dist_dn_matrix<r_t> max_values, const dist_dn_matrix<x_t> maxs) {
    assert(mat.n() == max_values.n() && max_values.shape() == maxs.shape() && maxs.m() == ctx.size());
    for (std::size_t i = 0; i < ctx.size(); i++)
        max_row_indices_equal(ctx[i], mat[i], max_values[i], maxs[i], i);
}

template <typename r_t, typename x_t>
void max_row_indices(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dist_dn_matrix<x_t> maxs) {
    assert(mat.n() == maxs.n() && maxs.m() == ctx.size());
    
    dist_dn_matrix<r_t> max_values(ctx, maxs.shape());
    max_rows(ctx, mat, max_values);

    max_row_indices_equal(ctx, mat, max_values, maxs);

    CHECK_NCCL( ncclGroupStart() );
    for (std::size_t i = 0; i < ctx.size(); i++)
        CHECK_NCCL( ncclAllReduce(maxs[i].buffer(), maxs[i].buffer(), maxs[i].size(), get_nccl_data_type<x_t>(), ncclMax, ctx(i), ctx[i].cuda_streams->get()) );
    CHECK_NCCL( ncclGroupEnd() );
}

template <typename r_t, typename x_t>
void index_log_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const dn_matrix<r_t> values) {
    ctx.set();
    assert(mat.n() == indices.n()  && indices.m() == 1 && mat.n() == values.n() && values.m() == 1);
    index_log_rows(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), values.buffer(), mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void index_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const dn_matrix<r_t> values) {
    ctx.set();
    assert(mat.n() == indices.n()  && indices.m() == 1 && mat.n() == values.n() && values.m() == 1);
    index_rows(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), values.buffer(), mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void index_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const dn_matrix<r_t> values, std::size_t rank) {
    ctx.set();
    assert(mat.n() == indices.n()  && indices.m() == 1 && mat.n() == values.n() && values.m() == 1);
    index_rows(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), values.buffer(), rank, mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void index_rows(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const dist_dn_matrix<r_t> values) {
    assert(mat.n() == indices.n() && indices.m() == 1 && mat.n() == values.n() && values.m() == ctx.size());
    for (std::size_t i = 0; i < ctx.size(); i++)
        index_rows(ctx[i], mat[i], indices, values[i], i);
    
    CHECK_NCCL( ncclGroupStart() );
    for (std::size_t i = 0; i < ctx.size(); i++)
        CHECK_NCCL( ncclAllReduce(values[i].buffer(), values[i].buffer(), values[i].size(), get_nccl_data_type<r_t>(), ncclSum, ctx(i), ctx[i].cuda_streams->get()) );
    CHECK_NCCL( ncclGroupEnd() );
}

template <typename r_t, typename x_t>
void add_indexed_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const r_t alpha) {
    ctx.set();
    assert(mat.n() == indices.n() && indices.m() == 1);
    add_indexed_rows(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), alpha, mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void add_indexed_rows_scale_all(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const r_t alpha, const r_t scalar) {
    ctx.set();
    assert(mat.n() == indices.n() && indices.m() == 1);
    add_indexed_rows_scale_all(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), alpha, mat.size(), mat.m(), scalar);
}

template <typename r_t, typename x_t>
void add_indexed_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const r_t alpha, const std::size_t rank, const std::size_t num_procs) {
    ctx.set();
    assert(mat.n() == indices.n() && indices.m() == 1);
    add_indexed_rows(ctx.cuda_streams->get(), mat.buffer(), indices.buffer(), alpha, rank, mat.size(), mat.m());
}

template <typename r_t, typename x_t>
void add_indexed_rows(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dn_matrix<x_t> indices, const r_t alpha) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        add_indexed_rows(ctx[i], mat[i], indices, alpha, i, ctx.size());
}

template <typename r_t, typename x_t>
void is_equal(const context ctx, const dn_matrix<x_t> mat1, const dn_matrix<x_t> mat2, const dn_matrix<r_t> out) {
    ctx.set();
    assert(mat1.shape() == mat2.shape() && mat1.shape() == out.shape());
    is_equal(ctx.cuda_streams->get(), mat1.buffer(), mat2.buffer(), out.buffer(), mat1.size());
}

template <typename r_t>
void subtract_rows(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<r_t> scalar) {
    ctx.set();
    assert(mat.n() == scalar.n() && scalar.m() == 1);
    subtract_rows(ctx.cuda_streams->get(), mat.buffer(), scalar.buffer(), mat.size(), mat.m());
}

template <typename r_t>
void subtract_rows_exp(const context ctx, const dn_matrix<r_t> mat, const dn_matrix<r_t> scalar, const dn_matrix<r_t> out) {
    ctx.set();
    assert(mat.n() == scalar.n() && scalar.m() == 1);
    assert(mat.shape() == out.shape());
    subtract_rows_exp(ctx.cuda_streams->get(), mat.buffer(), scalar.buffer(), out.buffer(), mat.size(), mat.m());
}

template <typename r_t>
void subtract_rows(const dist_context ctx, const dist_dn_matrix<r_t> mat, const dist_dn_matrix<r_t> scalar) {
    for (std::size_t i = 0; i < ctx.size(); i++)
        subtract_rows(ctx[i], mat[i], scalar[i]);
}

template <typename r_t>
void subtract(const context ctx, const dn_matrix<r_t> mat, r_t scalar) {
    ctx.set();
    subtract(ctx.cuda_streams->get(), mat.buffer(), scalar, mat.size());
}

template <typename r_t>
void axpby(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B, const r_t alpha, const r_t beta) {
    assert(A.shape() == B.shape());
    ctx.set();
    axpby(ctx.cuda_streams->get(), A.buffer(), B.buffer(), alpha, beta, A.size());
}

template <typename r_t, template<typename T> class dn_t>
void axpby(const dist_context ctx, const dn_t<r_t> A, const dn_t<r_t> B, const r_t alpha, const r_t beta) {
    assert(A.shape() == B.shape());
    for (std::size_t i = 0; i < ctx.size(); i++)    
        axpby(ctx[i], A[i], B[i], alpha, beta);
}

template <typename r_t>
void aaxpby(const context ctx, const dn_matrix<r_t> A, const dn_matrix<r_t> B, const r_t alpha, const r_t beta) {
    assert(A.shape() == B.shape());
    ctx.set();
    aaxpby(ctx.cuda_streams->get(), A.buffer(), B.buffer(), alpha, beta, A.size());
}

template <typename r_t, template<typename T> class dn_t>
void aaxpby(const dist_context ctx, const dn_t<r_t> A, const dn_t<r_t> B, const r_t alpha, const r_t beta) {
    assert(A.shape() == B.shape());
    for (std::size_t i = 0; i < ctx.size(); i++)
        aaxpby(ctx[i], A[i], B[i], alpha, beta);
}

template <typename r_t>
void adam_final(const context ctx, const dn_matrix<r_t> param, const dn_matrix<r_t> m, const dn_matrix<r_t> v, const r_t lr, const r_t c1, const r_t c2, const r_t eps) {
    assert(param.shape() == m.shape() && v.shape() == m.shape());
    ctx.set();
    adam_final(ctx.cuda_streams->get(), param.buffer(), m.buffer(), v.buffer(), lr, c1, c2, eps, param.size());
}

template <typename r_t, template<typename T> class dn_t>
void adam_final(const dist_context ctx, const dn_t<r_t> param, const dn_t<r_t> m, const dn_t<r_t> v, const r_t lr, const r_t c1, const r_t c2, const r_t eps) {
    assert(param.shape() == m.shape() && v.shape() == m.shape());
    for (std::size_t i = 0; i < ctx.size(); i++)
        adam_final(ctx[i], param[i], m[i], v[i], lr, c1, c2, eps);
}

template <typename r_t>
void transpose_helper_encoder(const context ctx, const dn_matrix<r_t> in, const dn_matrix<r_t> out, const std::size_t num_procs) {
    ctx.set();
    assert(in.size() == out.size());
    transpose_helper_encoder(ctx.cuda_streams->get(), in.buffer(), out.buffer(), in.size(), in.m(), num_procs);
}