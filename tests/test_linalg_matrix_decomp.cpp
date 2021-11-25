// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

TEST(TestMatrixDecomposition, TestQR)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // clang-format off
    std::vector<double> data = {
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0
    }; // clang-format on

    Matrix<double> ans(data, 3, 3);
    Matrix<double> a(data, 3, 3);
    Matrix<double> q(a.rows(), a.cols());
    Matrix<double> r(a.rows(), a.cols());

    qr(a.view(), q.view(), r.view());
    auto res = q * r;
    for (std::size_t i = 0; i < ans.rows(); ++i) {
        for (std::size_t j = 0; j < ans.cols(); ++j) {
            EXPECT_DOUBLE_EQ(res(i, j), ans(i, j));
        }
    }
}
