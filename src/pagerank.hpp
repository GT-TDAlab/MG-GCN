#pragma once

#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

#include <cuda_runtime_api.h>

#include "gcn.hpp"
#include "matrix.hpp"

template <typename x_t, typename v_t, typename r_t>
auto pagerank(const context ctx, csr_matrix<x_t, v_t, r_t> A, const r_t dump = 0.85, const r_t eps = 0.0001) {
    A.normalize(false);
    A = A.transpose();
    dn_matrix<r_t> p(A.n(), 1);
    std::fill(p.buffer(), p.buffer() + p.n(), 1);

    gcn_layer layer("0_", A, A.transpose(), 1, 1, false);
    layer.b()[0] = 1 - dump;
    layer.W()[0] = dump;

    auto reducer = [](auto a, auto b) {
        return std::max(a, std::abs(b));
    };

    for (auto p_prev = p;; p_prev = p.copy(ctx)) {
        p = layer(ctx, p);
        cudaDeviceSynchronize();
        auto err = std::transform_reduce(p.begin(), p.end(), p_prev.begin(), (r_t)0, reducer, std::minus<>());
        if (err < eps)
            break;
    }

    const auto K = p.n() / std::accumulate(p.begin(), p.end(), (r_t)0);
    std::for_each(p.begin(), p.end(), [=](auto &x) {
        x *= K;
    });

    return p;
} 