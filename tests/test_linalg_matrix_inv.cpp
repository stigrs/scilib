// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <cmath>
#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestInv)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> a_data = {
         1.0, 5.0,  4.0,  2.0,
        -2.0, 3.0,  6.0,  4.0,
         5.0, 1.0,  0.0, -1.0,
         2.0, 3.0, -4.0,  0.0
    };

    // Numpy:
    std::vector<double> ainv_data = {
        -0.19008264,  0.16528926,  0.28099174,  0.05785124,
         0.34710744, -0.21487603, -0.16528926,  0.02479339,
         0.16528926, -0.0785124,   0.01652893, -0.20247934,
        -0.60330579,  0.61157025,  0.23966942,  0.31404959
    };
    // clang-format on
    using extents_type = typename Matrix<double>::extents_type;

    Matrix<double> a(extents_type(4, 4), a_data);
    Matrix<double> ans(extents_type(4, 4), ainv_data);

    auto res = inv(a);

    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-8);
        }
    }
}

TEST(TestLinalg, TestInvColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> a_data = {
         1.0, 5.0,  4.0,  2.0,
        -2.0, 3.0,  6.0,  4.0,
         5.0, 1.0,  0.0, -1.0,
         2.0, 3.0, -4.0,  0.0};

    // Numpy:
    std::vector<double> ainv_data = {
        -0.19008264,  0.16528926,  0.28099174,  0.05785124,
         0.34710744, -0.21487603, -0.16528926,  0.02479339,
         0.16528926, -0.0785124,   0.01652893, -0.20247934,
        -0.60330579,  0.61157025,  0.23966942,  0.31404959
    };
    // clang-format on
    using extents_type = typename Matrix<double, stdex::layout_left>::extents_type;

    Matrix<double, stdex::layout_left> a(extents_type(4, 4), a_data);
    Matrix<double> ans_t(extents_type(4, 4), ainv_data);
    Matrix<double, stdex::layout_left> ans(transposed(ans_t).to_mdspan());

    auto res = inv(a);

    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-8);
        }
    }
}

TEST(TestLinalg, TestNotInvertible)
{
     using namespace Sci;
     using namespace Sci::Linalg;

     Matrix<double> A = {{1.0, 1.0, -2.0}, {1.0, -2.0, 1.0}, {-2.0, 1.0, 1.0}};
     EXPECT_ANY_THROW(inv(A));
}