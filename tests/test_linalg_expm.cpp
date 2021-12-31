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

TEST(TestLinalg, TestExpm)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // Example from Matlab:
    // clang-format off
    std::vector<double> ans_data = {
        2.7183, 1.7183, 1.0862,
        0.0000, 1.0000, 1.2642,
        0.0000, 0.0000, 0.3679
    };
    std::vector<double> A_data = {
        1.0, 1.0, 0.0, 
        0.0, 0.0, 2.0, 
        0.0, 0.0, -1.0
    };
    // clang-format off
    Matrix<double> ans(ans_data, 3, 3);
    Matrix<double> A(A_data, 3, 3);

    auto res = expm(A);

    for (std::size_t i = 0; i < res.extent(0); ++i) {
        for (std::size_t j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-4);
        }
    }
}

TEST(TestLinalg, TestExpmColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // Example from Matlab:
    // clang-format off
    std::vector<double> ans_data = {
        2.7183, 0.0000, 0.0000,
        1.7183, 1.0000, 0.0000,
        1.0862, 1.2642, 0.3679
    };
    std::vector<double> A_data = {
        1.0, 0.0,  0.0,
        1.0, 0.0,  0.0,
        0.0, 2.0, -1.0
    };
    // clang-format off
    Matrix<double, stdex::layout_left> ans(ans_data, 3, 3);
    Matrix<double, stdex::layout_left> A(A_data, 3, 3);

    auto res = expm(A);

    for (std::size_t j = 0; j < res.extent(1); ++j) {
        for (std::size_t i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-4);
        }
    }
}
