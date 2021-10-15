#include "test.hpp"

#include <iostream>
#include <filesystem>
#include <cstdint>
#include <vector>
#include <numeric>
#include <functional>

#include "matrix.hpp"
#include "gcn.hpp"
#include "pagerank.hpp"


int pagerank_test(const context ctx) {
    int pass = 0;

    csr_matrix<std::uint32_t, std::uint32_t, float> A("../test/data/cora_v2/graph.bin");
    auto p = pagerank(ctx, A);

    dn_matrix<float> pr("../test/data/cora_v2/pagerank.bin");

    auto reducer = [](auto a, auto b) {
        return std::max(a, std::abs(b));
    };

    auto transformer = [](auto a, auto b) {
        return (a - b) / b;
    };

    const auto max_err = std::transform_reduce(p.begin(), p.end(), pr.begin(), (float)0, reducer, transformer);
    ASSERT(max_err < 0.01);

    return pass;
}

int test_gcn_forward(const context ctx, std::filesystem::path dir) {
    int pass;
    
    csr_matrix<unsigned, unsigned, float> A(dir / "graph.bin");
    dn_matrix<float> X(dir / "features.bin");
    dn_matrix<std::int32_t> Y(dir / "labels.bin");
    dn_matrix<std::int32_t> S(dir / "sets.bin");

    const auto num_labels = 1 + *std::max_element(Y.begin(), Y.end());

    std::vector<std::pair<dn_matrix<float>, dn_matrix<float>>> weights;
    weights.emplace_back(dir / "0.bin", dir / "1.bin");
    weights.emplace_back(dir / "2.bin", dir / "3.bin");
    weights.emplace_back(dir / "4.bin", dir / "5.bin");
    
    std::vector<gcn_layer<unsigned, unsigned, float>> layers;
    std::vector<std::size_t> sizes =  { X.m(), 16, 16, Y.m() };
    A.normalize(true);
    auto A_t = A.transpose();
    for (std::size_t i = 1; i < sizes.size(); i++)
        layers.emplace_back(std::to_string(i - 1) + "_", A_t, A, sizes[i - 1], sizes[i], i + 1 < sizes.size());
    for (std::size_t i = 0; i < layers.size(); i++) {
        std::copy(weights[i].first.begin(), weights[i].first.end(), layers[i].W().begin());
        std::copy(weights[i].second.begin(), weights[i].second.end(), layers[i].b().begin());
    }

    std::vector<dn_matrix<float>> logits = { X };
    for (auto &layer: layers)
        logits.push_back(layer(ctx, logits.back()));

    for (int i = 0; i < 3; i++) {
        dn_matrix<float> logits_expected(dir / ("o" + std::to_string(i) + ".bin"));
        auto logit = logits[i + 1];
        // for (std::size_t i = 0; i < logit.n() * logit.m(); i++)
        //     ASSERT_CLOSE(logit[i], logits_expected[i]);
    }

	softmax_cross_entropy_loss<float, std::int32_t> loss_layer(std::to_string(sizes.size() - 1) + "_");
    auto [loss, acc] = loss_layer(ctx, logits.back(), Y);
    cudaDeviceSynchronize();
    std::cerr << loss << ' ' << acc << std::endl;
    auto G = loss_layer.backward();
    for (auto l_it = layers.rbegin(); l_it != layers.rend(); l_it++)
        G = l_it->backward(ctx, G);
    cudaDeviceSynchronize();

    for (int i = 2; i >= 0; i--) {
        dn_matrix<float> Gw(dir / ("g" + std::to_string(2 * i) + ".bin"));
        auto GW = layers[i].GW();
        for (std::size_t i = 0; i < Gw.n() * Gw.m(); i++)
            std::cerr << i << ' ' << Gw[i] << ' ' << GW[i] << std::endl;

        dn_matrix<float> Gb(dir / ("g" + std::to_string(2 * i + 1) + ".bin"));
        auto GB = layers[i].Gb();
        for (std::size_t i = 0; i < Gb.n() * Gb.m(); i++)
            std::cerr << i << ' ' << Gb[i] << ' ' << GB[i] << std::endl;
    }

    return pass;
}

int test_cross_entropy(const context ctx) {
    int pass = 0;
    dn_matrix<std::int32_t> Y(3, 1);
    std::vector<std::int32_t> buffer_in = {0, 0, 1};
    std::copy(buffer_in.begin(), buffer_in.end(), Y.begin());
    dn_matrix<float> logits(3, 3);
    std::vector<float> logits_in = {2, 1, 2, 4, 2, 1, 1, -1, 0};
    std::copy(logits_in.begin(), logits_in.end(), logits.begin());
	softmax_cross_entropy_loss<float, std::int32_t> loss_layer("0_");
    auto [loss, acc] = loss_layer(ctx, logits, Y);
    auto G = loss_layer.backward();
    cudaDeviceSynchronize();
    ASSERT_CLOSE(loss, 1.146482);
    std::vector<float> expected = { -0.1925604,  0.0517875,  0.1407729, -0.0520684,  0.0380651,  0.0140034, 0.2217470, -0.3033231,  0.0815762 };
    for (int i = 0; i < 9; i++)
        ASSERT_CLOSE(G[i], expected[i]);
    return pass;
}


int test_leaky_relu(const context ctx) {
    int pass = 0;
    dn_matrix<std::int32_t> Y(3, 1);
    std::vector<std::int32_t> buffer_in = {0, 0, 1};
    std::copy(buffer_in.begin(), buffer_in.end(), Y.begin());
    dn_matrix<float> logits(3, 3);
    std::vector<float> logits_in = {2, 1, 2, 4, 2, 1, 1, -1, 0};
    std::copy(logits_in.begin(), logits_in.end(), logits.begin());
	softmax_cross_entropy_loss<float, std::int32_t> loss_layer("0_");
    dn_matrix<float> H(3, 3);
    leaky_relu_forward(ctx, logits, H);
    auto [loss, acc] = loss_layer(ctx, H, Y);
    auto G = loss_layer.backward();
    cudaDeviceSynchronize();
    leaky_relu_backward(ctx, logits, G, G);
    cudaDeviceSynchronize();
    ASSERT_CLOSE(loss, 0.8637248);
    std::vector<float> expected = { -0.1925604,  0.0517875,  0.1407729, -0.0520684,  0.0380651, 0.0140034, 0.1924448, -0.0026324,  0.0007080 };
    for (int i = 0; i < 9; i++)
        ASSERT_CLOSE(G[i], expected[i]);
    return pass;
}

int test_g(const context ctx) {
    int pass = 0;
    dn_matrix<float> A(2, 2);
    A.init({1, 0, 0.5, 0.5});
    dn_matrix<float> X(2, 3);
    X.init({4, 2, 1, 1, -1, 0});
    dn_matrix<float> W(3, 2);
    W.init({1, 2, -1, 0, 0.5, 1.5});
    dn_matrix<float> b(1, 2);
    b.init({1, 0.5});
    dn_matrix<std::int32_t> Y(2, 1);
    Y.init({0, 1});

    dn_matrix<float> XW(2, 2);
    matmul(ctx, X, W, XW, 1.f, 0.f);
    dn_matrix<float> AXW(2, 2);
    broadcast_rows(ctx, b, AXW);
    matmul(ctx, A, XW, AXW, 1.f, 1.f);
    dn_matrix<float> H(2, 2);
    leaky_relu_forward(ctx, AXW, H);
	softmax_cross_entropy_loss<float, std::int32_t> loss_layer("0_");
    dn_matrix<float> ones(1, 2), G_b(1, 2);
    ones.init({1, 1});
    auto [loss, acc] = loss_layer(ctx, H, Y);
    auto G = loss_layer.backward();
    leaky_relu_backward(ctx, AXW, G, G);
    matmul(ctx, ones, G, G_b, (float)1, (float)0);
    dn_matrix<float> G_XW(2, 2);
    dn_matrix<float> G_W(3, 2);
    dn_matrix<float> G_out(2, 3);
    matmul(ctx, A, G, G_XW, 1.f, 0.f, true);
    matmul(ctx, X, G_XW, G_W, 1.f, 0.f, true);
    matmul(ctx, G_XW, W, G_out, 1.f, 0.f, false, true);
    cudaDeviceSynchronize();
    ASSERT_CLOSE(loss, 3.2750449);
    std::vector<float> expected = { -0.4992494, 0.4992494, 0.0237129, -0.0237129 };
    for (int i = 0; i < 4; i++)
        ASSERT_CLOSE(G[i], expected[i]);

    expected = { -0.4755365, 0.4755365 };
    for (int i = 0; i < 2; i++)
        ASSERT_CLOSE(G_b[i], expected[i]);

    expected = { -1.9377153, 1.9377153, -0.9866424, 0.9866424, -0.4873929, 0.4873929 };
    for (int i = 0; i < 6; i++)
        ASSERT_CLOSE(G_W[i], expected[i]);

    expected = { 0.4873929, 0.4873929, 0.4873930, -0.0118565, -0.0118565, -0.0118565 };
    for (int i = 0; i < 6; i++)
        ASSERT_CLOSE(G_out[i], expected[i]);

    return pass;
}

int test_csr_g(const context ctx) {
    int pass = 0;
    csr_matrix<unsigned, unsigned, float> A({0, 1, 3}, {0, 0, 1}, {1, 0.5, 0.5}, 2);
    dn_matrix<float> X(2, 3);
    X.init({4, 2, 1, 1, -1, 0});
    dn_matrix<float> W(3, 2);
    W.init({1, 2, -1, 0, 0.5, 1.5});
    dn_matrix<float> b(1, 2);
    b.init({1, 0.5});
    dn_matrix<std::int32_t> Y(2, 1);
    Y.init({0, 1});

    dn_matrix<float> XW(2, 2);
    matmul(ctx, X, W, XW, 1.f, 0.f);
    dn_matrix<float> AXW(2, 2);
    broadcast_rows(ctx, b, AXW);
    auto ext_buffer = get_matmul_buffer(ctx, A, XW, AXW, 1.f, 0.f);
    matmul(ctx, A, XW, AXW, ext_buffer, 1.f, 1.f);
    dn_matrix<float> H(2, 2);
    leaky_relu_forward(ctx, AXW, H);
	softmax_cross_entropy_loss<float, std::int32_t> loss_layer("0_");
    dn_matrix<float> ones(1, 2), G_b(1, 2);
    ones.init({1, 1});
    auto [loss, acc] = loss_layer(ctx, H, Y);
    auto G = loss_layer.backward();
    leaky_relu_backward(ctx, AXW, G, G);
    matmul(ctx, ones, G, G_b, (float)1, (float)0);
    dn_matrix<float> G_XW(2, 2);
    dn_matrix<float> G_W(3, 2);
    dn_matrix<float> G_out(2, 3);
    auto A_t = A.transpose();
    auto G_ext_buffer = get_matmul_buffer(ctx, A_t, G, G_XW, 1.f, 0.f);
    matmul(ctx, A_t, G, G_XW, G_ext_buffer, 1.f, 0.f);
    matmul(ctx, X, G_XW, G_W, 1.f, 0.f, true);
    matmul(ctx, G_XW, W, G_out, 1.f, 0.f, false, true);
    cudaDeviceSynchronize();
    ASSERT_CLOSE(loss, 3.2750449);
    std::vector<float> expected = { -0.4992494, 0.4992494, 0.0237129, -0.0237129 };
    for (int i = 0; i < 4; i++)
        ASSERT_CLOSE(G[i], expected[i]);

    expected = { -0.4755365, 0.4755365 };
    for (int i = 0; i < 2; i++)
        ASSERT_CLOSE(G_b[i], expected[i]);

    expected = { -1.9377153, 1.9377153, -0.9866424, 0.9866424, -0.4873929, 0.4873929 };
    for (int i = 0; i < 6; i++)
        ASSERT_CLOSE(G_W[i], expected[i]);

    expected = { 0.4873929, 0.4873929, 0.4873930, -0.0118565, -0.0118565, -0.0118565 };
    for (int i = 0; i < 6; i++)
        ASSERT_CLOSE(G_out[i], expected[i]);

    return pass;
}

int main() {
    int pass = 0;

    const auto ctx = context(0);
    // TEST(pagerank_test);
    // TEST_WARGS(test_gcn_forward, "../test/data/cora_v2/");
    TEST_WARGS(test_cross_entropy, ctx);
    TEST_WARGS(test_leaky_relu, ctx);
    TEST_WARGS(test_g, ctx);
    TEST_WARGS(test_csr_g, ctx);

    return pass;
}