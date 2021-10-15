#include "test.hpp"

#include <filesystem>
#include <cstdint>
#include <vector>
#include <numeric>

#include "dist_matrix.hpp"
#include "cuda_utils.hpp"


int test_constructor_and_conversion(const dist_context ctx) {
    int pass = 0;

	dn_matrix<double> A(8, 8);
	std::iota(A.begin(), A.end(), 0.0);
	dist_dn_matrix Ad(ctx, A);

	for (std::size_t i = 0; i < A.size(); i++)
		ASSERT_EQ(5 + A[i], 5 + Ad(i));

	dn_matrix<double> B(8, 8);
	Ad.to_dn_matrix(ctx, B);

	for (std::size_t i = 0; i < A.size(); i++)
		ASSERT_EQ(5 + A[i], 5 + B[i]);

    return pass;
}

int test_dist_dn_transpose(const dist_context ctx, std::vector<float> A, const std::size_t n, const std::size_t m, std::vector<float> expected) {
    int pass = 0;

    dn_matrix<float> mat_A(n, m);
    std::copy(A.begin(), A.end(), mat_A.begin());

    auto A_T = mat_A.transpose(ctx[0]);
    ctx[0].sync();

    dist_dn_matrix Ad(ctx, mat_A);
    auto Ad_T = Ad.transpose(ctx);
    dn_matrix<float> Ad_Ts(A_T.shape());
    Ad_T.to_dn_matrix(ctx, Ad_Ts);

    ctx.sync();

    for (std::size_t i = 0; i < A_T.size(); i++)
        ASSERT_CLOSE(Ad_Ts[i], A_T[i]);

    return pass;
}

int main() {
    int pass = 0;

    auto ctxs = dist_context(2);
    TEST_WARGS(test_dist_dn_transpose, ctxs, {1, 2, 3, 4, 5, 6, 7, 8}, 2, 4, {1, 5, 2, 6, 3, 7, 4, 8});
	TEST_WARGS(test_constructor_and_conversion, ctxs);

    return pass;
}