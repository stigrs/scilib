// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(TestLinalg, TestInv)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

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

    Matrix<double> a(a_data, 4, 4);
    Matrix<double> ans(ainv_data, 4, 4);

    auto res = inv(a.view());

    for (std::size_t i = 0; i < res.extent(0); ++i) {
        for (std::size_t j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-8);
        }
    }
}
