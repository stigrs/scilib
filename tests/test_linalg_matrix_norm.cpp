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

TEST(TestLinalg, TestMatrixNorm)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // clang-format off
    std::vector<double> a_data = {
        -3.0, 5.0, 7.0,
         0.0, 2.0, 8.0
    };
    // clang-format on
    Matrix<double> A(a_data, 2, 3);

    EXPECT_EQ(matrix_norm(A.view(), '1'), 15.0);
    EXPECT_EQ(matrix_norm(A.view(), 'I'), 15.0);
}

TEST(TestLinalg, TestMatrixNormColMajor)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // clang-format off
    std::vector<double> a_data = {
        -3.0, 0.0,
         5.0, 2.0,
         7.0, 8.0
    };
    // clang-format on
    Matrix<double, stdex::layout_left> A(a_data, 2, 3);

    EXPECT_EQ(matrix_norm(A.view(), '1'), 15.0);
    EXPECT_EQ(matrix_norm(A.view(), 'I'), 15.0);
}
