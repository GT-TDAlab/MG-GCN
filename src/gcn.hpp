#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include <optional>

#include "matrix.hpp"
#include "dist_matrix.hpp"
#include "cuda_utils.hpp"

template <typename x_t, typename v_t, typename r_t>
class sparse_linear {
	using csr_t = csr_matrix<x_t, v_t, r_t>;
	using dn_t = dn_matrix<r_t>;

	std::string name;
	csr_t A, A_T;
	std::size_t M = 0, M2 = 0;
	cuda_ptr<char> ext_buffer, ext_buffer2;

public:
	
	sparse_linear(std::string name, csr_t A, csr_t A_T) : name(name), A(A), A_T(A_T) {}

	void operator() (context ctx, dn_t B, dn_t C, bool discard = true) {
		if (B.m() != M) {
			M = B.m();
			ext_buffer = get_matmul_buffer(ctx, A, B, C, (r_t)1, discard ? (r_t)0 : (r_t)1);
		}
		ctx.record(name + "0_0_matmul-spmm", 0);
		matmul(ctx, A, B, C, ext_buffer, (r_t)1, discard ? (r_t)0 : (r_t)1);
		ctx.record(name + "0_1_matmul-spmm", 0);
		ctx.register_timer(name + "0_matmul-spmm", name + "0_0_matmul-spmm", name + "0_1_matmul-spmm");
	}

	void backward(context ctx, dn_t G, dn_t G_out, bool discard = true) {
		if (G.m() != M2) {
			M2 = G.m();
			ext_buffer2 = get_matmul_buffer(ctx, A_T, G, G_out, (r_t)1, discard ? (r_t)0 : (r_t)1);
		}
		ctx.record(name + "1_0_matmul-spmm", 0);
		matmul(ctx, A_T, G, G_out, ext_buffer2, (r_t)1, discard ? (r_t)0 : (r_t)1);
		ctx.record(name + "1_1_matmul-spmm", 0);
		ctx.register_timer(name + "1_matmul-spmm", name + "1_0_matmul-spmm", name + "1_1_matmul-spmm");
	}
};

template <bool row_partition, typename x_t, typename v_t, typename r_t>
class dist_sparse_linear {
	using csr_t = std::conditional_t<row_partition, dist_row_csr_matrix<x_t, v_t, r_t>, dist_csr_matrix<x_t, v_t, r_t>>;
	using dn_t = std::conditional_t<row_partition, dist_row_dn_matrix<r_t>, dist_dn_matrix<r_t>>;

	std::string name;
	csr_t A, A_T;
	std::size_t M = 0, M2 = 0;
	std::vector<cuda_ptr<char>> ext_buffers, ext_buffers2;
	std::vector<cuda_ptr<r_t>> bcast_buffer, bcast_buffer2;
	std::vector<dn_t> B_bcast;
	std::vector<dn_t> G_bcast;

public:
	
	dist_sparse_linear(std::string name, csr_t A, csr_t A_T, std::vector<cuda_ptr<r_t>> bcast_buffer, std::vector<cuda_ptr<r_t>> bcast_buffer2) : name(name), A(A), A_T(A_T), bcast_buffer(bcast_buffer), bcast_buffer2(bcast_buffer2) {}

	void operator() (const dist_context ctx, dn_t B, dn_t C, bool discard = true) {
		if (B.m() != M) {
			M = B.m();
			ext_buffers = get_matmul_buffer(ctx, A, B, C, (r_t)1, discard ? (r_t)0 : (r_t)1);
			B_bcast = std::vector{dn_t(ctx, B.n(), B.m(), bcast_buffer), dn_t(ctx, B.n(), B.m(), bcast_buffer2)};
		}
		matmul(ctx, A, B, C, ext_buffers, B_bcast, (r_t)1, discard ? (r_t)0 : (r_t)1, name + "0_");
		// matmul(ctx, A, B, C, ext_buffers, B_bcast[0], (r_t)1, (r_t)0);
	}
 
	void backward(const dist_context ctx, dn_t G, dn_t G_out, bool discard = true) {
		if (G.m() != M2) {
			M2 = G.m();
			ext_buffers2 = get_matmul_buffer(ctx, A_T, G, G_out, (r_t)1, discard ? (r_t)0 : (r_t)1);
			G_bcast = std::vector{dn_t(ctx, G.n(), G.m(), bcast_buffer), dn_t(ctx, G.n(), G.m(), bcast_buffer2)};
		}
		matmul(ctx, A_T, G, G_out, ext_buffers2, G_bcast, (r_t)1, discard ? (r_t)0 : (r_t)1, name + "1_");
		// matmul(ctx, A_T, G, G_out, ext_buffers2, G_bcast[0], (r_t)1, (r_t)0);
	}
};

template <typename r_t>
class linear {
	using dn_t = dn_matrix<r_t>;

	std::string name;

	dn_t W, G_W, mW, vW;
	dn_t b, G_b, mb, vb;

	dn_t X;

	dn_t ones;

	bool backward_out;

	std::size_t step;

public:

	linear(std::string name, std::size_t in, std::size_t out, bool backward_out = true) : name(name), W(in, out), G_W(in, out), b(1, out), G_b(1, out), backward_out(backward_out) {
		W.init();
		b.init(std::sqrt((r_t)1.0 / 3));
	}

	void setX(dn_t new_X) {
		X = new_X;
	}

	void operator() (context ctx, dn_t X, dn_t XW, bool discard = true) {
		broadcast_rows(ctx, b, XW, discard);
		ctx.record(name + "0_0_matmul-gemm", 0);
		matmul(ctx, X, W, XW, (r_t)1, (r_t)1);
		ctx.record(name + "0_1_matmul-gemm", 0);
		ctx.register_timer(name + "0_matmul-gemm", name + "0_0_matmul-gemm", name + "0_1_matmul-gemm");
		this->X = X;
	}

	void backward(context ctx, dn_t G, dn_t G_out, bool discard = true) {
		if (ones.n() != 1 || ones.m() != G.n()) {
			ones = dn_matrix<r_t>(1, G.n());
			std::fill(ones.begin(), ones.end(), 1);
		}
		ctx.record(name + "1_0_matmul-gemm", 0);
		matmul(ctx, ones, G, G_b, (r_t)1, (r_t)0);
		ctx.record(name + "1_1_matmul-gemm", 0);
		matmul(ctx, X, G, G_W, (r_t)1, (r_t)0, true);
		ctx.record(name + "1_2_matmul-gemm", 0);
		if (backward_out)
			matmul(ctx, G, W, G_out, (r_t)1, discard ? (r_t)0 : (r_t)1, false, true);
		ctx.record(name + "1_3_matmul-gemm", 0);
		ctx.register_timer(name + "1_matmul-gemm", name + "1_0_matmul-gemm", name + "1_3_matmul-gemm");
	}

	void update(const context ctx, const r_t lr, const r_t weight_decay) {
		axpby(ctx, G_W, W, -lr, 1 - weight_decay);
		axpy(ctx, G_b, b, -lr);
	}

	void adam_update(context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		if (mW.shape() != W.shape()) {
			mW = dn_matrix<r_t>(W.shape());
			mW.zero(ctx);
			vW = dn_matrix<r_t>(W.shape());
			vW.zero(ctx);
			mb = dn_matrix<r_t>(b.shape());
			mb.zero(ctx);
			vb = dn_matrix<r_t>(b.shape());
			vb.zero(ctx);
			step = 0;
		}
		step += 1;
		const r_t bc1 = 1 - std::pow(beta1, step);
		const r_t bc2 = 1 - std::pow(beta2, step);

		ctx.record(name + "0_adam-update", 0);
		axpy(ctx, W, G_W, weight_decay);
		axpby(ctx, G_W, mW, 1 - beta1, beta1);
		axpby(ctx, G_b, mb, 1 - beta1, beta1);
		aaxpby(ctx, G_W, vW, 1 - beta2, beta2);
		aaxpby(ctx, G_b, vb, 1 - beta2, beta2);
		adam_final(ctx, W, mW, vW, lr, bc1, bc2, eps);
		adam_final(ctx, b, mb, vb, lr, bc1, bc2, eps);
		ctx.record(name + "1_adam-update", 0);
		ctx.register_timer(name + "adam-update", name + "0_adam-update", name + "1_adam-update");
	}

	auto get_b() {
		return b;
	}

	auto get_W() {
		return W;
	}

	auto get_G_W() {
		return G_W;
	}

	auto get_G_b() {
		return G_b;
	}
};

template <typename r_t>
class dist_row_linear {
	using dn_t = dist_row_dn_matrix<r_t>;
	using rdn_t = repl_dn_matrix<r_t>;

	std::string name;

	rdn_t W, G_W, mW, vW;
	rdn_t b, G_b, mb, vb;

	dn_t X;

	rdn_t ones;

	bool backward_out;

	std::size_t step;

public:

	dist_row_linear(const dist_context ctx, std::string name, std::size_t in, std::size_t out, bool backward_out = true) : name(name), W(ctx, in, out), G_W(ctx, in, out), b(ctx, 1, out), G_b(ctx, 1, out), backward_out(backward_out) {
		W.init(ctx);
		b.init(ctx, std::sqrt((r_t)1.0 / 3));
	}

	void setX(dn_t new_X) {
		X = new_X;
	}

	void operator() (dist_context ctx, dn_t X, dn_t XW, bool discard = true) {
		broadcast_rows(ctx, b, XW, discard);
		ctx.record(name + "0_0_matmul-gemm", 0);
		matmul(ctx, X, W, XW, (r_t)1, (r_t)1);
		ctx.record(name + "0_1_matmul-gemm", 0);
		ctx.register_timer(name + "0_matmul-gemm", name + "0_0_matmul-gemm", name + "0_1_matmul-gemm");
		this->X = X;
	}

	void backward(dist_context ctx, dn_t G, dn_t G_out, bool discard = true) {
		if (ones.size() != ctx.size()) {
			ones = rdn_t(ctx, 1, G.n() / ctx.size());
			for (std::size_t i = 0; i < ctx.size(); i++)
				std::fill(ones[i].begin(), ones[i].end(), 1);
		}
		ctx.record(name + "1_0_matmul-gemm", 0);
		for (std::size_t i = 0; i < ctx.size(); i++)
			matmul(ctx[i], ones[i], G[i], G_b[i], (r_t)1, (r_t)0);
		G_b.allreduce(ctx);
		ctx.record(name + "1_1_matmul-gemm", 0);
		matmul(ctx, X, G, G_W, (r_t)1, (r_t)0);
		ctx.record(name + "1_2_matmul-gemm", 0);
		if (backward_out)
			matmul(ctx, G, W, G_out, (r_t)1, discard ? (r_t)0 : (r_t)1, true);
		ctx.record(name + "1_3_matmul-gemm", 0);
		ctx.register_timer(name + "1_matmul-gemm", name + "1_0_matmul-gemm", name + "1_3_matmul-gemm");
	}

	void update(const dist_context ctx, const r_t lr, const r_t weight_decay) {
		axpby(ctx, G_W, W, -lr, 1 - weight_decay);
		axpy(ctx, G_b, b, -lr);
	}

	void adam_update(dist_context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		if (mW.size() != W.size()) {
			mW = rdn_t(ctx, W.shape());
			mW.zero(ctx);
			vW = rdn_t(ctx, W.shape());
			vW.zero(ctx);
			mb = rdn_t(ctx, b.shape());
			mb.zero(ctx);
			vb = rdn_t(ctx, b.shape());
			vb.zero(ctx);
			step = 0;
		}
		step += 1;
		const r_t bc1 = 1 - std::pow(beta1, step);
		const r_t bc2 = 1 - std::pow(beta2, step);

		ctx.record(name + "0_adam-update", 0);
		axpy(ctx, W, G_W, weight_decay);
		axpby(ctx, G_W, mW, 1 - beta1, beta1);
		axpby(ctx, G_b, mb, 1 - beta1, beta1);
		aaxpby(ctx, G_W, vW, 1 - beta2, beta2);
		aaxpby(ctx, G_b, vb, 1 - beta2, beta2);
		adam_final(ctx, W, mW, vW, lr, bc1, bc2, eps);
		adam_final(ctx, b, mb, vb, lr, bc1, bc2, eps);
		ctx.record(name + "1_adam-update", 0);
		ctx.register_timer(name + "adam-update", name + "0_adam-update", name + "1_adam-update");
	}

	auto get_b() {
		return b;
	}

	auto get_W() {
		return W;
	}

	auto get_G_W() {
		return G_W;
	}

	auto get_G_b() {
		return G_b;
	}
};

template <typename r_t>
class dist_linear {
	using dn_t = dist_dn_matrix<r_t>;

	dn_t W, G_W, mW, vW;
	dn_t b, G_b, mb, vb;

	dn_t X;

	dn_t ones;

	std::size_t step;

	std::vector<dn_matrix<r_t>> tA, tB, tC, tB2, tC2;

public:

	dist_linear(const dist_context ctx, std::size_t in, std::size_t out) : W(ctx, in, out), G_W(ctx, in, out), b(ctx, 1, out), G_b(ctx, 1, out) {
		W.init();
		b.init(std::sqrt((r_t)1.0 / 3));
	}

	void operator() (const dist_context ctx, dn_t X, dn_t XW) {
		broadcast_rows(ctx, b, XW);
		if (tB.size() != ctx.size()) {
			tB.clear();
			tC.clear();
			for (std::size_t i = 0; i < ctx.size(); i++) {
				ctx[i].set();
				tC.emplace_back(XW.n(), XW.m() / ctx.size(), cuda_malloc<r_t>(XW.n() * XW.m() / ctx.size()));
				tB.emplace_back(W.n() / ctx.size(), W.m() / ctx.size(), cuda_malloc<r_t>(W.n() / ctx.size() * W.m() / ctx.size()));
			}
		}
		matmul(ctx, X, W, XW, (r_t)1, (r_t)1, tB, tC);
		this->X = X;
	}

	void backward(const dist_context ctx, dn_t G, dn_t G_out) {
		if (ones.size() != ctx.size()) {
			ones = dn_t(ctx, 1, G.n() * ctx.size());
			for (std::size_t i = 0; i < ctx.size(); i++)
				std::fill(ones[i].begin(), ones[i].end(), 1);
		}
		for (std::size_t i = 0; i < ctx.size(); i++)
			matmul(ctx[i], ones[i], G[i], G_b[i], (r_t)1, (r_t)0);
		
		if (tA.size() != ctx.size()) {
			tA.clear();
			for (std::size_t i = 0; i < ctx.size(); i++) {
				ctx[i].set();
				tA.emplace_back(X[i].shape(), cuda_malloc<r_t>(X[i].size()));
			}
		}
		matmul(ctx, X, G, G_W, (r_t)1, (r_t)0, tA);

		if (tC2.size() != ctx.size()) {
			tB2.clear();
			tC2.clear();
			for (std::size_t i = 0; i < ctx.size(); i++) {
				ctx[i].set();
				tC2.emplace_back(G_out.n(), G_out.m() / ctx.size(), cuda_malloc<r_t>(G_out.n() * G_out.m() / ctx.size()));
				tB2.emplace_back(W.m() / ctx.size(), W.n() / ctx.size(), cuda_malloc<r_t>(W.m() / ctx.size() * W.n() / ctx.size()));
			}
		}
		matmul(ctx, G, W.transpose(ctx), G_out, (r_t)1, (r_t)0, tB2, tC2);
	}

	void update(const dist_context ctx, const r_t lr, const r_t weight_decay) {
		axpby(ctx, G_W, W, -lr, 1 - weight_decay);
		axpy(ctx, G_b, b, -lr);
	}

	void adam_update(const dist_context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		if (mW.size() != W.size()) {
			mW = dn_t(ctx, W.shape());
			mW.zero(ctx);
			vW = dn_t(ctx, W.shape());
			vW.zero(ctx);
			mb = dn_t(ctx, b.shape());
			mb.zero(ctx);
			vb = dn_t(ctx, b.shape());
			vb.zero(ctx);
			step = 0;
		}
		step += 1;
		const r_t bc1 = 1 - std::pow(beta1, step);
		const r_t bc2 = 1 - std::pow(beta2, step);
		axpy(ctx, W, G_W, weight_decay);
		axpby(ctx, G_W, mW, 1 - beta1, beta1);
		axpby(ctx, G_b, mb, 1 - beta1, beta1);
		aaxpby(ctx, G_W, vW, 1 - beta2, beta2);
		aaxpby(ctx, G_b, vb, 1 - beta2, beta2);
		adam_final(ctx, W, mW, vW, lr, bc1, bc2, eps);
		adam_final(ctx, b, mb, vb, lr, bc1, bc2, eps);
	}

	auto get_b() {
		return b;
	}

	auto get_W() {
		return W;
	}

	auto get_G_W() {
		return G_W;
	}

	auto get_G_b() {
		return G_b;
	}
};

template <typename x_t, typename v_t, typename r_t>
class gcn_layer {

	std::string name;

	sparse_linear<x_t, v_t, r_t> A;
	linear<r_t> lin;
	std::optional<linear<r_t>> res_lin;
	dn_matrix<r_t> HW; // HW_buffer
	cuda_ptr<r_t> AHW_buffer;
	dn_matrix<r_t> AHW; // AHW_buffer
	dn_matrix<r_t> G_HW; // HW_buffer
	dn_matrix<r_t> G_out; // AHW_buffer
	bool activation;

	bool residual_layer;
	bool backward_spmm;

	dn_matrix<r_t> H;

public:
	gcn_layer(std::string name, csr_matrix<x_t, v_t, r_t> A, csr_matrix<x_t, v_t, r_t> A_T, std::size_t in, std::size_t out, bool activation, bool residual_layer = false, bool backward_spmm = true, cuda_ptr<r_t> HW_buffer = nullptr) : 
		name(name), A(name, A, A_T), lin(name, in, out, backward_spmm), res_lin(in == out || !residual_layer ? std::nullopt : std::make_optional(linear<r_t>(name, in, out, backward_spmm))), HW(A.m(), std::min(in, out), HW_buffer), AHW_buffer(cuda_malloc_managed<r_t>(std::max(A.n() * out, A_T.n() * in))), AHW(A.n(), out, AHW_buffer),
		G_HW(A_T.n(), std::min(in, out), HW_buffer), G_out(A_T.n(), in, AHW_buffer), activation(activation), residual_layer(residual_layer), backward_spmm(backward_spmm) {}

	// H is m x in
	auto operator()(context ctx, dn_matrix<r_t> H) {
		this->H = H;
		if (HW.m() == AHW.m()) {
			lin(ctx, H, HW);
			A(ctx, HW, AHW);
		}
		else {
			A(ctx, H, HW);
			lin(ctx, HW, AHW);
		}
		if (activation) {
			ctx.record(name + "0_0_activation", 0);
			leaky_relu_forward(ctx, AHW, AHW);
			ctx.record(name + "0_1_activation", 0);
			ctx.register_timer(name + "0_activation", name + "0_0_activation", name + "0_1_activation");
		}
		if (res_lin)
			(*res_lin)(ctx, H, AHW, false);
		else if (residual_layer)
			axpy(ctx, H, AHW, (r_t)1);
		return AHW;
	}

	auto backward(context ctx, dn_matrix<r_t> G) {
		auto T = G;
		if (activation) {
			ctx.record(name + "1_0_activation", 0);
			leaky_relu_backward(ctx, AHW, G, AHW);
			ctx.record(name + "1_1_activation", 0);
			ctx.register_timer(name + "1_activation", name + "1_0_activation", name + "1_1_activation");
			T = AHW;
		}
		if (HW.m() == AHW.m()) {
			if (backward_spmm)
				A.backward(ctx, T, G_HW);
			else
				G_HW = T;
			lin.backward(ctx, G_HW, G_out);
		}
		else {
			lin.setX(H);
			lin.backward(ctx, T, G_HW);
			if (backward_spmm)
				A.backward(ctx, G_HW, G_out);
			else
				G_out = G_HW;
		}
		if (res_lin)
			res_lin->backward(ctx, G, G_out, false);
		else if (residual_layer)
			axpy(ctx, G, G_out, (r_t)1);
		return G_out;
	}

	void update(const context ctx, const r_t lr, const r_t weight_decay) {
		lin.update(ctx, lr, weight_decay);
		if (res_lin)
			res_lin->update(ctx, lr, weight_decay);
	}

	void adam_update(const context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		lin.adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
		if (res_lin)
			res_lin->adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
	}

	auto b() {
		return lin.get_b();
	}

	auto W() {
		return lin.get_W();
	}

	auto GW() {
		return lin.get_G_W();
	}

	auto Gb() {
		return lin.get_G_b();
	}
};

template <bool row_partition, typename x_t, typename v_t, typename r_t>
class dist_gcn_layer {

	std::string name;

	dist_sparse_linear<row_partition, x_t, v_t, r_t> A;
	std::conditional_t<row_partition, dist_row_linear<r_t>, dist_linear<r_t>> lin;
	std::optional<std::conditional_t<row_partition, dist_row_linear<r_t>, dist_linear<r_t>>> res_lin;
	using csr_t = std::conditional_t<row_partition, dist_row_csr_matrix<x_t, v_t, r_t>, dist_csr_matrix<x_t, v_t, r_t>>;
	using dn_t = std::conditional_t<row_partition, dist_row_dn_matrix<r_t>, dist_dn_matrix<r_t>>;
	
	std::vector<cuda_ptr<r_t>> AHW_buffer;
	dn_t HW; // HW_buffer
	dn_t AHW; // AHW_buffer
	dn_t G_HW; // HW_buffer
	dn_t G_out; // AHW_buffer
	bool activation;

	bool residual_layer;
	bool backward_spmm;

	dn_t H;

public:
	dist_gcn_layer(const dist_context ctx, std::string name, csr_t A, csr_t A_T, std::size_t in, std::size_t out, bool activation, bool residual_layer = false, bool backward_spmm = true, std::vector<cuda_ptr<r_t>> HW_buffer = {}, std::vector<cuda_ptr<r_t>> bcast_buffer = {}, std::vector<cuda_ptr<r_t>> bcast_buffer2 = {}) : 
		name(name), A(name, A, A_T, bcast_buffer, bcast_buffer2), lin(ctx, name, in, out, backward_spmm), res_lin(in == out || !residual_layer ? std::nullopt : std::make_optional(std::conditional_t<row_partition, dist_row_linear<r_t>, dist_linear<r_t>>(ctx, name, in, out, backward_spmm))), AHW_buffer([&]() {
			std::vector<cuda_ptr<r_t>> t;
			for (std::size_t i = 0; i < ctx.size(); i++) {
				ctx[i].set();
				t.emplace_back(cuda_malloc<r_t>(std::max(A.n() * out, A_T.n() * in) / ctx.size()));
			}
			return t;
		}()), HW(ctx, A.m(), std::min(in, out), HW_buffer), AHW(ctx, A.n(), out, AHW_buffer),
		G_HW(ctx, A_T.n(), std::min(in, out), HW_buffer), G_out(ctx, A_T.n(), in, AHW_buffer), activation(activation), residual_layer(residual_layer), backward_spmm(backward_spmm) {}

	// H is m x in
	auto operator()(dist_context ctx, dn_t H) {
		this->H = H;
		if (HW.m() == AHW.m()) {
			lin(ctx, H, HW);
			A(ctx, HW, AHW);
		}
		else {
			A(ctx, H, HW);
			lin(ctx, HW, AHW);
		}
		if (activation) {
			ctx.record(name + "0_0_activation", 0);
			leaky_relu_forward(ctx, AHW, AHW);
			ctx.record(name + "0_1_activation", 0);
			ctx.register_timer(name + "0_activation", name + "0_0_activation", name + "0_1_activation");
		}
		if (res_lin)
			(*res_lin)(ctx, H, AHW, false);
		else if (residual_layer)
			axpy(ctx, H, AHW, (r_t)1);
		return AHW;
	}

	auto backward(dist_context ctx, dn_t G) {
		auto T = G;
		if (activation) {
			ctx.record(name + "1_0_activation", 0);
			leaky_relu_backward(ctx, AHW, G, AHW);
			ctx.record(name + "1_1_activation", 0);
			ctx.register_timer(name + "1_activation", name + "1_0_activation", name + "1_1_activation");
			T = AHW;
		}
		if (HW.m() == AHW.m()) {
			if (backward_spmm)
				A.backward(ctx, T, G_HW);
			else
				G_HW = T;
			lin.backward(ctx, G_HW, G_out);
		}
		else {
			lin.setX(H);
			lin.backward(ctx, T, G_HW);
			if (backward_spmm)
				A.backward(ctx, G_HW, G_out);
			else
				G_out = G_HW;
		}
		if (res_lin)
			res_lin->backward(ctx, G, G_out, false);
		else if (residual_layer)
			axpy(ctx, G, G_out, (r_t)1);
		return G_out;
	}

	void update(const dist_context ctx, const r_t lr, const r_t weight_decay) {
		lin.update(ctx, lr, weight_decay);
		if (res_lin)
			res_lin->update(ctx, lr, weight_decay);
	}

	void adam_update(const dist_context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		lin.adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
		if (res_lin)
			res_lin->adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
	}

	auto b() {
		return lin.get_b();
	}

	auto W() {
		return lin.get_W();
	}

	auto GW() {
		return lin.get_G_W();
	}

	auto Gb() {
		return lin.get_G_b();
	}
};

template <typename r_t>
class softmax {

	dn_matrix<r_t> ones;
	dn_matrix<r_t> H;
	dn_matrix<r_t> H_R;
	dn_matrix<r_t> maxs;
	const bool copy;

public:
	softmax(bool copy = true) : copy(copy) {}
	 
	auto operator()(const context ctx, dn_matrix<r_t> temp) {
		if (copy) {
			if (!H.buffer())
				H = dn_matrix<r_t>(temp.n(), temp.m());
			temp.copy_to(ctx, H);
		}
		else
			H = temp;
		if (!maxs.buffer())
			maxs = dn_matrix<r_t>(H.n(), 1);
		max_rows(ctx, H, maxs);
		// subtract_rows(ctx, H, maxs);
		// exp(ctx, H, H);
		subtract_rows_exp(ctx, H, maxs, H);
		if (!ones.buffer()) {
			ones = dn_matrix<r_t>(H.m(), 1);
			std::fill(ones.begin(), ones.end(), 1);
		}
		if (!H_R.buffer()) {
			H_R = dn_matrix<r_t>(H.n(), 1);
		}
		matmul(ctx, H, ones, H_R, (r_t)1, (r_t)0);
		scale_rows(ctx, H, H_R);
		return H;
	}
};

template <typename r_t>
class dist_softmax {

	dist_dn_matrix<r_t> ones;
	dist_dn_matrix<r_t> H;
	dist_dn_matrix<r_t> H_R;
	dist_dn_matrix<r_t> maxs;
	const bool copy;

public:
	dist_softmax(bool copy = true) : copy(copy) {}
	 
	auto operator()(const dist_context ctx, dist_dn_matrix<r_t> temp) {
		if (copy) {
			if (H.size() == 0)
				H = dist_dn_matrix<r_t>(ctx, temp.shape());
			temp.copy_to(ctx, H);
		}
		else
			H = temp;
		if (maxs.size() == 0)
			maxs = dist_dn_matrix<r_t>(ctx, H.n(), ctx.size());
		max_rows(ctx, H, maxs);
		subtract_rows(ctx, H, maxs);
		exp(ctx, H, H);
		if (ones.size() == 0) {
			ones = dist_dn_matrix<r_t>(ctx, H.m() / ctx.size(), ctx.size());
			for (std::size_t i = 0; i < ones.size(); i++)
				std::fill(ones[i].begin(), ones[i].end(), 1);
		}
		if (H_R.size() == 0)
			H_R = dist_dn_matrix<r_t>(ctx, H.n(), ctx.size());
		for (std::size_t i = 0; i < ctx.size(); i++)
			matmul(ctx[i], H[i], ones[i], H_R[i], (r_t)1, (r_t)0);
		
		CHECK_NCCL( ncclGroupStart() );
        for (std::size_t i = 0; i < ctx.size(); i++)
            CHECK_NCCL( ncclAllReduce(H_R[i].buffer(), H_R[i].buffer(), H_R[i].size(), get_nccl_data_type<r_t>(), ncclSum, ctx(i), ctx[i].cuda_streams->get()) );
        CHECK_NCCL( ncclGroupEnd() );

		for (std::size_t i = 0; i < ctx.size(); i++)
			scale_rows(ctx[i], H[i], H_R[i]);
		return H;
	}
};

template <typename r_t>
class dist_row_softmax {

	using dn_t = dist_row_dn_matrix<r_t>;
	using rdn_t = repl_dn_matrix<r_t>;
	rdn_t ones;
	dn_t H;
	dn_t H_R;
	dn_t maxs;
	const bool copy;

public:
	dist_row_softmax(bool copy = true) : copy(copy) {}
	 
	auto operator()(const dist_context ctx, dn_t temp) {
		if (copy) {
			if (H.size() == 0)
				H = dn_t(ctx, temp.shape());
			temp.copy_to(ctx, H);
		}
		else
			H = temp;
		if (maxs.size() == 0)
			maxs = dn_t(ctx, H.n(), 1);
		for (std::size_t i = 0; i < ctx.size(); i++) {
			max_rows(ctx[i], H[i], maxs[i]);
			// subtract_rows(ctx[i], H[i], maxs[i]);
			subtract_rows_exp(ctx[i], H[i], maxs[i], H[i]);
		}
		// exp(ctx, H, H);
		if (ones.size() == 0) {
			ones = rdn_t(ctx, H.m(), 1);
			for (std::size_t i = 0; i < ones.size(); i++)
				std::fill(ones[i].begin(), ones[i].end(), 1);
		}
		if (H_R.size() == 0)
			H_R = dn_t(ctx, H.n(), 1);
		for (std::size_t i = 0; i < ctx.size(); i++) {
			matmul(ctx[i], H[i], ones[i], H_R[i], (r_t)1, (r_t)0);
			scale_rows(ctx[i], H[i], H_R[i]);
		}
		return H;
	}
};

template <typename r_t, typename x_t>
class softmax_cross_entropy_loss {

	std::string name;

	softmax<r_t> softmax_layer;
	dn_matrix<r_t> G;
	dn_matrix<r_t> L;
	dn_matrix<x_t> P;
	dn_matrix<r_t> T;
	cuda_ptr<r_t> loss, acc;

public:

	softmax_cross_entropy_loss(std::string name, bool copy = true) : name(name), softmax_layer(copy) {}

	auto operator()(context ctx, dn_matrix<r_t> H, dn_matrix<x_t> Y) {
		ctx.record(name + "0_loss-layer", 0);

		auto O = softmax_layer(ctx, H);

		if (!P.buffer())
			P = dn_matrix<x_t>(Y.shape());
		max_row_indices(ctx, O, P);

		if (!L.buffer())
			L = dn_matrix<r_t>(Y.shape());
		index_log_rows(ctx, O, Y, L);
		
		G = O;
		add_indexed_rows(ctx, G, Y, (r_t)-1);
		scale_mat(ctx, G, (r_t)1 / Y.n());
		// add_indexed_rows_scale_all(ctx, G, Y, (r_t)-1, (r_t)1 / Y.n());
		
		if (!T.buffer())
			T = dn_matrix<r_t>(Y.shape());
		is_equal(ctx, Y, P, T);
		if (!loss) {
			loss = cuda_malloc_managed<r_t>(1);
			acc = cuda_malloc_managed<r_t>(1);
		}
		abssum(ctx, L, loss[0]);
		abssum(ctx, T, acc[0]);

		ctx.record(name + "1_loss-layer", 0);
		ctx.register_timer(name + "loss-layer", name + "0_loss-layer", name + "1_loss-layer");

		ctx.sync();
		return std::make_pair(loss[0] / O.n(), acc[0] / O.n());
	}

	auto backward() {
		return G;
	}
};

template <typename r_t, typename x_t>
class dist_softmax_cross_entropy_loss {
private:
	dist_softmax<r_t> softmax_layer;
	dist_dn_matrix<r_t> G;
	dist_dn_matrix<r_t> L;
	dist_dn_matrix<x_t> P;
	dn_matrix<r_t> T;
	cuda_ptr<r_t> loss, acc;

public:

	dist_softmax_cross_entropy_loss(bool copy = true) : softmax_layer(copy) {}

	auto operator()(const dist_context ctx, dist_dn_matrix<r_t> H, dn_matrix<x_t> Y) {
		auto O = softmax_layer(ctx, H);
		if (G.size() == 0)
			G = dist_dn_matrix<r_t>(ctx, O.shape());
		
		O.copy_to(ctx, G);
		add_indexed_rows(ctx, G, Y, (r_t)-1);
		scale_mat(ctx, G, (r_t)1 / Y.n());
		if (P.size() == 0)
			P = dist_dn_matrix<x_t>(ctx, std::pair{Y.n(), ctx.size()});
		max_row_indices(ctx, O, P);
		log(ctx, O, O);
		if (L.size() == 0)
			L = dist_dn_matrix<r_t>(ctx, std::pair{Y.n(), ctx.size()});
		index_rows(ctx, O, Y, L);
		if (T.size() == 0)
			T = dn_matrix<r_t>(Y.shape());
		is_equal(ctx[0], Y, P[0], T);
		if (!loss) {
			loss = cuda_malloc_managed<r_t>(1);
			acc = cuda_malloc_managed<r_t>(1);
		}
		abssum(ctx[0], L[0], loss[0]);
		abssum(ctx[0], T, acc[0]);
		ctx[0].sync();
		return std::make_pair(loss[0] / O.n(), acc[0] / O.n());
	}

	auto backward() {
		return G;
	}
};

template <typename r_t, typename x_t>
class dist_row_softmax_cross_entropy_loss {
private:
	dist_row_softmax<r_t> softmax_layer;
	using dn_t = dist_row_dn_matrix<r_t>;

	std::string name;
	
	dn_t G;
	dn_t L;
	dist_row_dn_matrix<x_t> P;
	dn_t T;
	std::vector<cuda_ptr<r_t>> losses, accs;

public:

	dist_row_softmax_cross_entropy_loss(std::string name, bool copy = true) : name(name), softmax_layer(copy) {}

	auto operator()(dist_context ctx, dn_t H, dist_row_dn_matrix<x_t> Y) {
		ctx.record(name + "0_loss-layer", 0);
		
		auto O = softmax_layer(ctx, H);

		if (P.size() == 0)
			P = dist_row_dn_matrix<x_t>(ctx, std::pair{Y.n(), 1});
		for (std::size_t i = 0; i < ctx.size(); i++)
			max_row_indices(ctx[i], O[i], P[i]);
		
		if (L.size() == 0)
			L = dn_t(ctx, std::pair{Y.n(), 1});
		for (std::size_t i = 0; i < ctx.size(); i++)
			index_log_rows(ctx[i], O[i], Y[i], L[i]);

		G = O;
		for (std::size_t i = 0; i < ctx.size(); i++)
			add_indexed_rows(ctx[i], G[i], Y[i], (r_t)-1);
		scale_mat(ctx, G, (r_t)1 / Y.n());

		if (T.size() == 0)
			T = dn_t(ctx, Y.shape());
		for (std::size_t i = 0; i < ctx.size(); i++)
			is_equal(ctx[i], Y[i], P[i], T[i]);

		while (losses.size() < ctx.size()) {
			losses.emplace_back(cuda_malloc_managed<r_t>(1));
			accs.emplace_back(cuda_malloc_managed<r_t>(1));
		}

		for (std::size_t i = 0; i < ctx.size(); i++) {
			abssum(ctx[i], L[i], losses[i][0]);
			abssum(ctx[i], T[i], accs[i][0]);
		}

		ctx.record(name + "1_loss-layer", 0);
		ctx.register_timer(name + "loss-layer", name + "0_loss-layer", name + "1_loss-layer");

		ctx.sync();
		return std::make_pair(std::accumulate(losses.begin(), losses.end(), (r_t)0, [](auto a, auto b) {return a + b[0];}) / O.n(), std::accumulate(accs.begin(), accs.end(), (r_t)0, [](auto a, auto b) {return a + b[0];}) / O.n());
	}

	auto backward() {
		return G;
	}
};

template <typename x_t, typename v_t, typename r_t>
class gcn {

	std::vector<gcn_layer<x_t, v_t, r_t>> layers_;
	softmax_cross_entropy_loss<r_t, std::int32_t> loss_layer;
	cuda_ptr<r_t> HW_buffer;

public:

	gcn(csr_matrix<x_t, v_t, r_t> A, std::vector<std::size_t> sizes, bool residual_layer = false) : loss_layer(std::to_string(sizes.size() - 1) + "_", residual_layer) {
		A.normalize(true);
		auto A_T = A.transpose();
		std::size_t max_d = 0;
		for (std::size_t i = 0; i < sizes.size() - 1; i++)
			max_d = std::max(max_d, std::min(sizes[i], sizes[i + 1]));
		HW_buffer = cuda_malloc_managed<r_t>(std::max(A.n(), A.m()) * max_d);
		for (std::size_t i = 1; i < sizes.size(); i++)
			layers_.emplace_back(std::to_string(i - 1) + "_", A_T, A, sizes[i - 1], sizes[i], i + 1 < sizes.size(), residual_layer, i != 1, HW_buffer);
	}

	gcn(csr_matrix<x_t, v_t, r_t> A, std::vector<std::size_t> sizes, std::vector<std::pair<dn_matrix<r_t>, dn_matrix<r_t>>> weights) : gcn(A, sizes) {
		assert(layers_.size() == weights.size());
		for (std::size_t i = 0; i < layers_.size(); i++) {
			std::copy(weights[i].first.begin(), weights[i].first.end(), layers_[i].W().begin());
			std::copy(weights[i].second.begin(), weights[i].second.end(), layers_[i].b().begin());
		}
	}

	auto operator()(const context ctx, dn_matrix<r_t> H) {
		for (auto &layer: layers_)
			H = layer(ctx, H);
		return H;
	}

	auto train_forward(const context ctx, dn_matrix<r_t> H, dn_matrix<std::int32_t> Y) {
		H = operator()(ctx, H);
		return loss_layer(ctx, H, Y);
	}

	auto backward(const context ctx) {
		auto G = loss_layer.backward();
		for (auto l_it = layers_.rbegin(); l_it != layers_.rend(); l_it++)
			G = l_it->backward(ctx, G);
	}

	auto update(const context ctx, const r_t lr, const r_t weight_decay) {
		for (auto &layer: layers_)
			layer.update(ctx, lr, weight_decay);
	}

	auto adam_update(const context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		for (auto &layer: layers_)
			layer.adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
	}

	const auto & layers() const {
		return layers_;
	}
};

template <bool row_partition, typename x_t, typename v_t, typename r_t>
class dist_gcn {

	std::vector<dist_gcn_layer<row_partition, x_t, v_t, r_t>> layers_;
	std::conditional_t<row_partition, dist_row_softmax_cross_entropy_loss<r_t, std::int32_t>, dist_softmax_cross_entropy_loss<r_t, std::int32_t>> loss_layer;
	std::vector<cuda_ptr<r_t>> HW_buffer;
	std::vector<cuda_ptr<r_t>> bcast_buffer;
	std::vector<cuda_ptr<r_t>> bcast_buffer2;

	using csr_t = std::conditional_t<row_partition, dist_row_csr_matrix<x_t, v_t, r_t>, dist_csr_matrix<x_t, v_t, r_t>>;
	using dn_t = std::conditional_t<row_partition, dist_row_dn_matrix<r_t>, dist_dn_matrix<r_t>>;
	using idn_t = std::conditional_t<row_partition, dist_row_dn_matrix<std::int32_t>, dn_matrix<std::int32_t>>;

public:

	dist_gcn(const dist_context ctx, csr_t A, csr_t A_T, std::vector<std::size_t> sizes, bool residual_layer = false) : loss_layer(std::to_string(sizes.size() - 1) + "_", residual_layer) {
		std::size_t max_d = 0;
		for (std::size_t i = 0; i < sizes.size() - 1; i++)
			max_d = std::max(max_d, std::min(sizes[i], sizes[i + 1]));
		for (std::size_t i = 0; i < ctx.size(); i++) {
			ctx[i].set();
			HW_buffer.emplace_back(cuda_malloc<r_t>(std::max(A.n(), A.m()) * max_d / ctx.size()));
			bcast_buffer.emplace_back(cuda_malloc<r_t>(std::max(A.n(), A.m()) * max_d / ctx.size()));
			bcast_buffer2.emplace_back(cuda_malloc<r_t>(std::max(A.n(), A.m()) * max_d / ctx.size()));
		}
		for (std::size_t i = 1; i < sizes.size(); i++)
			layers_.emplace_back(ctx, std::to_string(i - 1) + "_", A_T, A, sizes[i - 1], sizes[i], i + 1 < sizes.size(), residual_layer, i != 1, HW_buffer, bcast_buffer, bcast_buffer2);
	}

	auto operator()(const dist_context ctx, dn_t H) {
		for (auto &layer: layers_)
			H = layer(ctx, H);
		return H;
	}

	auto train_forward(const dist_context ctx, dn_t H, idn_t Y) {
		H = operator()(ctx, H);
		return loss_layer(ctx, H, Y);
	}

	auto backward(const dist_context ctx) {
		auto G = loss_layer.backward();
		for (auto l_it = layers_.rbegin(); l_it != layers_.rend(); l_it++)
			G = l_it->backward(ctx, G);
	}

	auto update(const dist_context ctx, const r_t lr, const r_t weight_decay) {
		for (auto &layer: layers_)
			layer.update(ctx, lr, weight_decay);
	}

	auto adam_update(const dist_context ctx, const r_t lr, const r_t beta1, const r_t beta2, const r_t weight_decay, const r_t eps) {
		for (auto &layer: layers_)
			layer.adam_update(ctx, lr, beta1, beta2, weight_decay, eps);
	}

	const auto & layers() const {
		return layers_;
	}
};