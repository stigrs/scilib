// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalg, TestLstsq)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // Example from Intel MKL:
    // clang-format off
    std::vector<double> xans_data = {
        -0.69, -0.24,  0.06,
        -0.80, -0.08,  0.21,
         0.38,  0.12, -0.65,
         0.29, -0.24,  0.42,
         0.29,  0.35, -0.30
    };
    std::vector<double> a_data = {
         0.12, -8.19,  7.69, -2.26, -4.71,
        -6.91,  2.22, -5.12, -9.08,  9.96,
        -3.33, -8.94, -6.72, -4.40, -9.98,
         3.97,  3.33, -2.74, -7.92, -3.20
    };
    std::vector<double> b_data = {
         7.30,  0.47, -6.28,
         1.33,  6.58, -3.42,
         2.68, -1.71,  3.46,
        -9.62, -0.79,  0.41,
         0.00,  0.00,  0.00
    };
    // clang-format on
    Matrix<double> xans(xans_data, 5, 3);
    Matrix<double> a(a_data, 4, 5);
    Matrix<double> b(b_data, 5, 3);

    lstsq(a.view(), b.view());

    for (std::size_t i = 0; i < b.extent(0); ++i) {
        for (std::size_t j = 0; j < b.extent(1); ++j) {
            EXPECT_NEAR(b(i, j), xans(i, j), 5.0e-3);
        }
    }
}