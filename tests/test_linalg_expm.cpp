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
    Matrix<double> ans(stdex::dextents<Sci::index, 2>(3, 3), ans_data);
    Matrix<double> A(stdex::dextents<Sci::index, 2>(3, 3), A_data);

    auto res = expm(A);

    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
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
    Matrix<double, stdex::layout_left> ans(stdex::dextents<Sci::index, 2>(3, 3), ans_data);
    Matrix<double, stdex::layout_left> A(stdex::dextents<Sci::index, 2>(3, 3), A_data);

    auto res = expm(A);

    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-4);
        }
    }
}
