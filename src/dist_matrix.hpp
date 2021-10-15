#pragma once

#include <cassert>
#include <vector>
#include <type_traits>
#include <future>

#include <nccl.h>

#include "matrix.hpp"

class dist_context {
	std::vector<context> contexts;
	std::vector<ncclComm_t> comms;

public:

	const bool overlap;

	int bcast_stream_id() const {
		return overlap ? 1 : 0;
	}

	dist_context() = default;

	dist_context(std::size_t P, bool overlap = true) : overlap(overlap) {
		for (std::size_t i = 0; i < P; i++)
			contexts.emplace_back(i);
		comms.resize(P);
		CHECK_NCCL( ncclCommInitAll(comms.data(), P, nullptr) );
	}

	auto size() const {
		return contexts.size();
	}

	void sync() const {
		for (const auto &ctx: contexts)
			ctx.sync();
	}

	auto operator[] (std::size_t i) const {
		return contexts[i];
	}

	auto operator() (std::size_t i) const {
		return comms[i];
	}

	void record(const std::string str, const std::size_t stream_id) {
		for (std::size_t i = 0; i < size(); i++)
			contexts[i].record(str, stream_id);
	}

	void wait(const std::string str, const std::size_t stream_id) {
		for (std::size_t i = 0; i < size(); i++)
			contexts[i].wait(str, stream_id);
	}

	void register_timer(const std::string str, const std::string beg, const std::string end) {
        for (std::size_t i = 0; i < size(); i++)
			contexts[i].register_timer(str, beg, end);
    }

	void register_walltime(const std::string str, std::size_t stream_id) {
        for (std::size_t i = 0; i < size(); i++)
			contexts[i].register_walltime(str, stream_id);
    }

    auto measure_walltime(const std::string str) {
		std::vector<float> times;
		times.reserve(size());
		for (std::size_t i = 0; i < size(); i++)
			times.emplace_back(contexts[i].measure_walltime(str));
		return times;
    }

	auto measure(const std::string str) const {
        std::vector<float> times;
		times.reserve(size());
		for (std::size_t i = 0; i < size(); i++)
			times.emplace_back(contexts[i].measure(str));
		return times;
    }

	void dump_timers(std::ostream &out, std::string prefix = "") {
		for (std::size_t i = 0; i < size(); i++)
			contexts[i].dump_timers(out, prefix + std::to_string(i) + "_");
	}
};

template <typename r_t>
void subtract(r_t *mat, r_t scalar, const std::size_t size);

template <typename x_t, typename v_t, typename r_t>
class dist_csr_matrix {
	using matrix_t = csr_matrix<x_t, v_t, r_t>;

	const dist_context ctx;
	std::vector<matrix_t> As;
	std::vector<std::shared_ptr<matrix_t>> bAs;

public:

	auto n() const {
		v_t N = 0;
		for (const auto &A: As)
			N += A.n();
		return N;
	}

	auto m() const {
		return As[0].m();
	}

	auto size() const {
		return As.size();
	}

	auto operator() (std::size_t i) const {
		return As[i];
	}

	auto operator[] (std::size_t i) const {
		return *bAs[i];
	}

	void bcast(std::size_t i) const {
		ctx[i].set();

		CHECK_NCCL( ncclGroupStart() );
		for (std::size_t j = 0; j < size(); j++) {
			CHECK_NCCL( ncclBroadcast(As[i].indptr_.get(), bAs[j]->indptr_.get(), As[i].n() + 1, get_nccl_data_type<x_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
			CHECK_NCCL( ncclBroadcast(As[i].indices_.get() + As[i].indptr_[0], bAs[j]->indices_.get(), As[i].nnz(), get_nccl_data_type<v_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
			CHECK_NCCL( ncclBroadcast(As[i].data_.get() + As[i].indptr_[0], bAs[j]->data_.get(), As[i].nnz(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
		}
		CHECK_NCCL( ncclGroupEnd() );

		for (std::size_t j = 0; j < size(); j++) {
			ctx[j].sync();
			bAs[j]->N_ = As[i].n();
			ctx[j].set();
			subtract(bAs[j]->indptr_.get(), bAs[j]->indptr_[0], bAs[j]->N_ + 1);
		}

		for (std::size_t j = 0; j < size(); j++) {
			ctx[j].sync();
			bAs[j]->init_mat();
		}
	}

	dist_csr_matrix(const dist_context ctx, const matrix_t A, const std::vector<v_t> p) : ctx(ctx) {
        std::size_t ptr_max = 0, ind_max = 0;
		for (std::size_t i = 0; i < p.size() - 1; i++) {
            As.emplace_back(cuda_ptr<x_t>(A.indptr_, A.indptr_.get() + p[i]), A.indices_, A.data_, p[i + 1] - p[i], A.M_);
			ptr_max = std::max(ptr_max, (std::size_t)(p[i + 1] - p[i]));
			ind_max = std::max(ind_max, std::size_t(A.indptr_[p[i + 1]] - A.indptr_[p[i]]));
		}
		for (std::size_t i = 0; i < size(); i++) {
			CHECK_CUDA( cudaSetDevice(i) );
			// change N each time
			auto indptr = cuda_malloc_managed<x_t>(ptr_max + 1);
			indptr[ptr_max] = indptr[0] = 0;
			bAs.emplace_back(std::make_shared<matrix_t>(indptr, cuda_malloc_managed<v_t>(ind_max), cuda_malloc_managed<r_t>(ind_max), ptr_max, A.M_));
		}
	}

};

template <typename x_t, typename v_t, typename r_t>
class dist_row_csr_matrix {
	using matrix_t = csr_matrix<x_t, v_t, r_t>;

	const dist_context ctx;
	std::vector<std::vector<matrix_t>> As;

	template <typename Iter, typename Int>
    std::size_t lowerbound_index(Iter begin, Iter end, Int v) {
        return std::distance(begin + 1, std::lower_bound(begin, end, v + 1));
    }

	template <typename Int>
    std::size_t lowerbound_index(const std::vector<Int> &v_arr, Int v) {
        return lowerbound_index(v_arr.cbegin(), v_arr.cend(), v);
    }

public:

	auto n() const {
		std::size_t N = 0;
		for (const auto &A: As)
			N += A[0].n();
		return N;
	}

	auto m() const {
		std::size_t M = 0;
		for (const auto &A: As[0])
			M += A.m();
		return M;
	}

	auto size() const {
		return As.size();
	}

	auto size(std::size_t i) const {
		return As[i].size();
	}

	auto operator[] (std::pair<std::size_t, std::size_t> ij) const {
		return As[ij.first][ij.second];
	}

	dist_row_csr_matrix(const dist_context ctx, const matrix_t A, const std::vector<v_t> p, const std::vector<v_t> q) : ctx(ctx) {
		auto [indptr, indices, data] = A.buffer();
		for (std::size_t i = 0; i < p.size() - 1; i++) {
			std::vector<cuda_ptr<x_t>> indptrs;
			for (std::size_t j = 0; j < q.size() - 1; j++) {
				indptrs.emplace_back(cuda_malloc_managed<x_t>(p[i + 1] - p[i] + 1));
				std::fill_n(exec_policy, indptrs.back().get(), p[i + 1] - p[i] + 1, 0);
			}
			std::for_each(exec_policy, &indptr[p[i]], &indptr[p[i + 1]], [&](const auto &indptr_i) {
				const auto idx = &indptr_i - indptr.get();
				for (auto k = indptr[idx]; k < indptr[idx + 1]; k++) {
					auto jdx = indices[k];
					auto d = data[k];
					auto j = lowerbound_index(q, jdx);
					indptrs[j][idx - p[i] + 1]++;
				}
			});
			std::vector<cuda_ptr<v_t>> indicess;
			std::vector<cuda_ptr<r_t>> datas;
			for (auto &ptr: indptrs) {
				std::inclusive_scan(exec_policy, ptr.get(), ptr.get() + p[i + 1] - p[i] + 1, ptr.get());
				indicess.emplace_back(cuda_malloc_managed<v_t>(ptr[p[i + 1] - p[i]]));
				datas.emplace_back(cuda_malloc_managed<r_t>(ptr[p[i + 1] - p[i]]));
			}
			std::vector<std::shared_ptr<x_t[]>> f;
			for (std::size_t j = 0; j < q.size() - 1; j++) {
				f.emplace_back(std::shared_ptr<x_t[]>(new x_t[p[i + 1] - p[i]]));
				std::fill_n(exec_policy, f.back().get(), p[i + 1] - p[i], 0);
			}
			std::for_each(exec_policy, &indptr[p[i]], &indptr[p[i + 1]], [&](const auto &indptr_i) {
				const auto idx = &indptr_i - indptr.get();
				for (auto k = indptr[idx]; k < indptr[idx + 1]; k++) {
					auto jdx = indices[k];
					auto d = data[k];
					auto j = lowerbound_index(q, jdx);
					indicess[j][indptrs[j][idx - p[i]] + f[j][idx - p[i]]] = jdx - q[j];
					datas[j][indptrs[j][idx - p[i]] + f[j][idx - p[i]]] = d;
					f[j][idx - p[i]]++;
				}
			});
			As.emplace_back(0);
			for (std::size_t j = 0; j < q.size() - 1; j++)
				As.back().emplace_back(indptrs[j], indicess[j], datas[j], p[i + 1] - p[i], q[j + 1] - q[j]);
		}
	}
};

template <typename r_t>
void axpy(const context ctx, const r_t *A, const r_t *B, const std::size_t size, const r_t alpha, const int inc_A = 1, const int inc_B = 1);

template <typename r_t>
class dist_dn_matrix {
    using matrix_t = dn_matrix<r_t>;

    std::vector<matrix_t> As;

public:

	dist_dn_matrix() = default;

	auto n() const {
		return As[0].n();
	}
	
	auto m() const {
		return As.size() * As[0].m();
	}

	auto shape() const {
		return std::make_pair(n(), m());
	}

	auto size() const {
		return As.size();
	}

	auto operator[] (std::size_t i) const {
		return As[i];
	}

	auto operator() (std::size_t i) const {
		return As[(i / As[0].m()) % size()][i / m() * As[0].m() + i % As[0].m()];
	}

	dist_dn_matrix(const dist_context ctx, std::size_t N, std::size_t M) {
		assert(M % ctx.size() == 0);
		for (std::size_t i = 0; i < ctx.size(); i++)
			As.emplace_back(N, M / ctx.size());
	}

	dist_dn_matrix(const dist_context ctx, std::pair<std::size_t, std::size_t> shape) : dist_dn_matrix(ctx, shape.first, shape.second) {}

    dist_dn_matrix(const dist_context ctx, const matrix_t A) : dist_dn_matrix(ctx, A.N_, A.M_) {
		ctx.sync();
		for (std::size_t row = 0; row < n(); row++)
			for (std::size_t i = 0; i < ctx.size(); i++) {
				std::size_t col_begin = i * m() / size(), col_end = (i + 1) * m() / size();
				std::copy(A.buffer() + row * m() + col_begin, A.buffer() + row * m() + col_end, As[i].buffer() + row * As[i].m());
			}
	}

	void to_dn_matrix(const dist_context ctx, matrix_t A) const {
		ctx.sync();
		for (std::size_t row = 0; row < n(); row++)
			for (std::size_t i = 0; i < ctx.size(); i++) {
				std::size_t col_begin = i * m() / size();
				std::copy(As[i].buffer() + row * As[i].m(), As[i].buffer() + (row + 1) * As[i].m(), A.buffer() + row * m() + col_begin);
			}
	}

	auto part2d(std::size_t offset, std::size_t N) const {
		std::vector<matrix_t> AAs;
		for (std::size_t i = 0; i < As.size(); i++)
			AAs.emplace_back(N, As[i].m(), cuda_ptr<r_t>(As[i].buffer_, As[i].buffer() + offset * As[i].m()));
		return AAs;
	}

	void init(r_t gain = std::sqrt(2 / (1 + 0.01 * 0.01))) {
		for (auto &A: As)
			A.init(gain);
	}

	void set_bcast_buffer(std::vector<cuda_ptr<r_t>> bcast_buffer) {
		// TODO
	}

	void zero(const dist_context ctx) {
		for (std::size_t i = 0; i < ctx.size(); i++)
			As[i].zero(ctx[i]);
	}

	auto copy(const dist_context ctx, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		dist_dn_matrix clone(ctx, n(), m());
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], clone[i], copy_type);
		return clone;
	}

	void copy_to(const dist_context ctx, const dist_dn_matrix other, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], other[i], copy_type);
	}

	auto transpose(const dist_context ctx) const {
		timer t("Transpose", &ctx);
		std::vector<matrix_t> Ts, Es;
		for (std::size_t i = 0; i < ctx.size(); i++) {
			Ts.emplace_back(As[i].transpose(ctx[i]));
			Es.emplace_back(Ts.back().shape());
		}

		{
			timer t("Transpose encoder", &ctx);
			for (std::size_t i = 0; i < ctx.size(); i++)
				transpose_helper_encoder(ctx[i], Ts[i], Es[i], ctx.size());
		}

		{
			timer t("Transpose alltoall", &ctx);		
			CHECK_NCCL( ncclGroupStart() );
			for (std::size_t i = 0; i < ctx.size(); i++) {
				for (std::size_t j = 0; j < ctx.size(); j++) {
					CHECK_NCCL( ncclSend(Es[i].buffer() + j * Es[i].size() / ctx.size(), Es[i].size() / ctx.size(), get_nccl_data_type<r_t>(), j, ctx(i), ctx[i].cuda_streams->get()) );
					CHECK_NCCL( ncclRecv(Ts[j].buffer() + i * Ts[i].size() / ctx.size(), Es[i].size() / ctx.size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
				}
			}
			CHECK_NCCL( ncclGroupEnd() );
		}

		dist_dn_matrix clone(ctx, m(), n());

		for (std::size_t i = 0; i < ctx.size(); i++)
			clone.As[i] = matrix_t(m(), n() / ctx.size(), Ts[i].shared_buffer());
		
		return clone;
	}

};

template <typename r_t>
class dist_row_dn_matrix {
    using matrix_t = dn_matrix<r_t>;

    std::vector<matrix_t> As;

public:

	dist_row_dn_matrix() = default;

	auto n() const {
		std::size_t N = 0;
		for (const auto &A: As)
			N += A.n();
		return N;
	}
	
	auto m() const {
		return As[0].m();
	}

	auto shape() const {
		return std::make_pair(n(), m());
	}

	auto size() const {
		return As.size();
	}

	auto operator[] (std::size_t i) const {
		return As[i];
	}

	dist_row_dn_matrix(const dist_context ctx, std::size_t N, std::size_t M, std::vector<cuda_ptr<r_t>> buffer = {}) {
		assert(N % ctx.size() == 0);
		for (std::size_t i = 0; i < ctx.size(); i++)
			As.emplace_back(N / ctx.size(), M, i < buffer.size() ? buffer[i] : nullptr);
	}

	dist_row_dn_matrix(const std::vector<std::size_t> p, std::size_t M, std::vector<cuda_ptr<r_t>> buffer = {}) {
		for (std::size_t i = 0; i < p.size() - 1; i++)
			As.emplace_back(p[i + 1] - p[i], M, i < buffer.size() ? buffer[i] : nullptr);
	}

	dist_row_dn_matrix(const dist_context ctx, std::pair<std::size_t, std::size_t> shape) : dist_row_dn_matrix(ctx, shape.first, shape.second) {}

    dist_row_dn_matrix(const dist_context ctx, const matrix_t A) : dist_row_dn_matrix(ctx, A.N_, A.M_) {
		ctx.sync();
		auto A_buffer = A.buffer();
		for (std::size_t i = 0; i < ctx.size(); i++) {
			std::copy(exec_policy, A_buffer, A_buffer + As[i].size(), As[i].buffer());
			A_buffer += As[i].size();
		}
	}

	void to_dn_matrix(const dist_context ctx, matrix_t A) const {
		ctx.sync();
		auto A_buffer = A.buffer();
		for (std::size_t i = 0; i < ctx.size(); i++) {
			std::copy(exec_policy, As[i].buffer(), As[i].buffer() + As[i].size(), A_buffer);
			A_buffer += As[i].size();
		}
	}

	void bcast(const dist_context ctx, std::size_t i, dist_row_dn_matrix bAs, int stream_id = 1) {
		// assert(!streams || ctx.size() == streams.size());
		
		ctx[i].set();

		CHECK_NCCL( ncclGroupStart() );
		for (std::size_t j = 0; j < size(); j++)
			CHECK_NCCL( ncclBroadcast(As[i].buffer(), bAs[j].buffer(), As[i].size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams[stream_id].get()) );
		CHECK_NCCL( ncclGroupEnd() );
	}

	// auto part2d(std::size_t offset, std::size_t N) const {
	// 	std::vector<matrix_t> AAs;
	// 	for (std::size_t i = 0; i < As.size(); i++)
	// 		AAs.emplace_back(N, As[i].m(), cuda_ptr<r_t>(As[i].buffer_, As[i].buffer() + offset * As[i].m()));
	// 	return AAs;
	// }

	void init(r_t gain = std::sqrt(2 / (1 + 0.01 * 0.01))) {
		for (auto &A: As)
			A.init(gain / std::sqrt((r_t)size()));
	}

	void zero(const dist_context ctx) {
		for (std::size_t i = 0; i < ctx.size(); i++)
			As[i].zero(ctx[i]);
	}

	auto copy(const dist_context ctx, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		dist_row_dn_matrix clone(ctx, n(), m());
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], clone[i], copy_type);
		return clone;
	}

	void copy_to(const dist_context ctx, const dist_row_dn_matrix other, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], other[i], copy_type);
	}

	// auto transpose(const dist_context ctx) const {
	// 	timer t("Transpose", &ctx);
	// 	std::vector<matrix_t> Ts, Es;
	// 	for (std::size_t i = 0; i < ctx.size(); i++) {
	// 		Ts.emplace_back(As[i].transpose(ctx[i]));
	// 		Es.emplace_back(Ts.back().shape());
	// 	}

	// 	{
	// 		timer t("Transpose encoder", &ctx);
	// 		for (std::size_t i = 0; i < ctx.size(); i++)
	// 			transpose_helper_encoder(ctx[i], Ts[i], Es[i], ctx.size());
	// 	}

	// 	{
	// 		timer t("Transpose alltoall", &ctx);		
	// 		CHECK_NCCL( ncclGroupStart() );
	// 		for (std::size_t i = 0; i < ctx.size(); i++) {
	// 			for (std::size_t j = 0; j < ctx.size(); j++) {
	// 				CHECK_NCCL( ncclSend(Es[i].buffer() + j * Es[i].size() / ctx.size(), Es[i].size() / ctx.size(), get_nccl_data_type<r_t>(), j, ctx(i), ctx[i].cuda_streams->get()) );
	// 				CHECK_NCCL( ncclRecv(Ts[j].buffer() + i * Ts[i].size() / ctx.size(), Es[i].size() / ctx.size(), get_nccl_data_type<r_t>(), i, ctx(j), ctx[j].cuda_streams->get()) );
	// 			}
	// 		}
	// 		CHECK_NCCL( ncclGroupEnd() );
	// 	}

	// 	dist_dn_matrix clone(ctx, m(), n());

	// 	for (std::size_t i = 0; i < ctx.size(); i++)
	// 		clone.As[i] = matrix_t(m(), n() / ctx.size(), Ts[i].shared_buffer());
		
	// 	return clone;
	// }

};

template <typename r_t>
class repl_dn_matrix {
    using matrix_t = dn_matrix<r_t>;

    std::vector<matrix_t> As;

public:

	repl_dn_matrix() = default;

	auto n() const {
		return As[0].n();
	}
	
	auto m() const {
		return As[0].m();
	}

	auto shape() const {
		return std::make_pair(n(), m());
	}

	auto size() const {
		return As.size();
	}

	auto operator[] (std::size_t i) const {
		return As[i];
	}

	auto operator() (std::size_t i) const {
		return As[0][i];
	}

	repl_dn_matrix(const dist_context ctx, std::size_t N, std::size_t M) {
		for (std::size_t i = 0; i < ctx.size(); i++)
			As.emplace_back(N, M);
	}

    repl_dn_matrix(const dist_context ctx, const matrix_t A) : repl_dn_matrix(ctx, A.N_, A.M_) {
		ctx.sync();

		CHECK_NCCL( ncclGroupStart() );
		for (std::size_t j = 0; j < size(); j++)
			CHECK_NCCL( ncclBroadcast(A.buffer(), As[j].buffer(), n() * m(), get_nccl_data_type<r_t>(), 0, ctx(j), ctx[j].cuda_streams->get()) );
		CHECK_NCCL( ncclGroupEnd() );
	}

	void to_dn_matrix(const dist_context ctx, matrix_t A) const {
		ctx.sync();
		A = As[0];
	}

	void allreduce(const dist_context ctx) const {
		CHECK_NCCL( ncclGroupStart() );
		for (std::size_t i = 0; i < ctx.size(); i++)
			CHECK_NCCL( ncclAllReduce(As[i].buffer(), As[i].buffer(), As[i].size(), get_nccl_data_type<r_t>(), ncclSum, ctx(i), ctx[i].cuda_streams->get()) );
		CHECK_NCCL( ncclGroupEnd() );
	}

	auto part2d(std::size_t offset, std::size_t N) const {
		std::vector<matrix_t> AAs;
		for (std::size_t i = 0; i < As.size(); i++)
			AAs.emplace_back(N, As[i].m(), cuda_ptr<r_t>(As[i].buffer_, As[i].buffer() + offset * As[i].m()));
		return AAs;
	}

	void init(const dist_context ctx, r_t gain = std::sqrt(2 / (1 + 0.01 * 0.01))) {
		if (As.size()) {
			As[0].init(gain);
			CHECK_NCCL( ncclGroupStart() );
			for (std::size_t j = 0; j < size(); j++)
				CHECK_NCCL( ncclBroadcast(As[0].buffer(), As[j].buffer(), n() * m(), get_nccl_data_type<r_t>(), 0, ctx(j), ctx[j].cuda_streams->get()) );
			CHECK_NCCL( ncclGroupEnd() );
		}
	}

	void zero(const dist_context ctx) {
		for (std::size_t i = 0; i < ctx.size(); i++)
			As[i].zero(ctx[i]);
	}

	auto copy(const dist_context ctx, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		repl_dn_matrix clone(ctx, n(), m());
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], clone[i], copy_type);
		return clone;
	}

	void copy_to(const dist_context ctx, const repl_dn_matrix other, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
		for (std::size_t i = 0; i < As.size(); i++)
			As[i].copy_to(ctx[i], other[i], copy_type);
	}

	auto transpose(const dist_context ctx) const {
		timer t("Transpose", &ctx);

		repl_dn_matrix clone(ctx, m(), n());

		for (std::size_t i = 0; i < ctx.size(); i++)
			clone.As[i] = As[i].transpose();
		
		return clone;
	}

};