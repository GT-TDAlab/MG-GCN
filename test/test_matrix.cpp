#include "test.hpp"

#include <filesystem>
#include <cstdint>
#include <vector>

#include "matrix.hpp"
#include "cuda_utils.hpp"


int read_binary_graph_test() {
    int pass = 0;

    csr_matrix<std::uint32_t, std::uint32_t, float> A("../test/data/toyA/graph.bin");

    ASSERT_EQ(A.n(), 4);
    ASSERT_EQ(A.m(), 4);
    ASSERT_EQ(A.nnz(), 8);

    csr_matrix<std::uint32_t, std::uint32_t, float> B("../test/data/toyB/graph.bin");

    ASSERT_EQ(B.n(), 4);
    ASSERT_EQ(B.m(), 4);
    ASSERT_EQ(B.nnz(), 12);

    return pass;
}

int read_binary_features_test() {
    int pass = 0;

    dn_matrix<float> A("../test/data/toyA/features.bin");

    ASSERT_EQ(A.n(), 4);
    ASSERT_EQ(A.m(), 2);

    dn_matrix<float> B("../test/data/toyA/features.bin");

    ASSERT_EQ(B.n(), 4);
    ASSERT_EQ(B.m(), 2);

    return pass;
}

int read_reddit_dataset_test() {
    int pass = 0;

    if (std::filesystem::exists("../test/data/reddit/graph.bin")) {
        csr_matrix<std::uint32_t, std::uint32_t, float> R("../test/data/reddit/graph.bin");
        ASSERT_EQ(R.n(), 232968);
        ASSERT_EQ(R.m(), 232968);
        ASSERT_EQ(R.nnz(), 114848860);
    }

    if (std::filesystem::exists("../test/data/reddit/features.bin")) {
        dn_matrix<float> R("../test/data/reddit/features.bin");
        ASSERT_EQ(R.n(), 232968);
        ASSERT_EQ(R.m(), 608);
    }

    return pass;
}

int csr_to_dn_test() {
    int pass = 0;

    csr_matrix<std::uint32_t, std::uint32_t, float> A("../test/data/toyA/graph.bin");

    const std::vector<float> expected = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
    const auto A_dn = A.as_dn();
    
    for (std::size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(A_dn[i], expected[i]);

    return pass;
}

int test_dn_transpose(std::vector<float> A, const std::size_t n, const std::size_t m, std::vector<float> expected) {
    int pass = 0;

    const auto ctx = context(0);
    dn_matrix<float> mat_A(n, m);
    std::copy(A.begin(), A.end(), mat_A.begin());

    auto A_T = mat_A.transpose(ctx);
    cudaDeviceSynchronize();
    for (std::size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(A_T[i], expected[i]);

    return pass;
}

int test_csr_transpose() {
    int pass = 0;

    csr_matrix<std::uint32_t, std::uint32_t, float> A("../test/data/toyA/graph.bin");
    auto B = A.transpose();

    const auto A_dn = A.as_dn();
    const auto B_dn = B.as_dn();
    
    const auto ctx = context(0);
    const auto B_dn_cand = A_dn.transpose(ctx);
    cudaDeviceSynchronize();
    for (std::size_t i = 0; i < B_dn.n() * B_dn.m(); i++)
        ASSERT_EQ(B_dn[i], B_dn_cand[i]);

    return pass;
}

int main() {
    int pass = 0;

    TEST(read_binary_graph_test);
    TEST(read_binary_features_test);
    TEST(read_reddit_dataset_test);
    TEST(csr_to_dn_test);
    TEST_WARGS(test_dn_transpose, {1, 2, 3, 4, 5, 6, 7, 8}, 2, 4, {1, 5, 2, 6, 3, 7, 4, 8});
    TEST(test_csr_transpose);

    return pass;
}