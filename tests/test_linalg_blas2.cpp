// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>

TEST(TestLinalg, TestMatrixVectorProduct)
{
    std::vector<int> va = {1, -1, 2, 0, -3, 1};
    Scilib::Matrix<int> a(va, 2, 3);
    Scilib::Vector<int> x(std::vector<int>{2, 1, 0}, 3);
    Scilib::Vector<int> y(std::vector<int>{1, -3}, 2);
    EXPECT_EQ((a * x), y);
}

#ifdef USE_MKL
// Does not work with OpenBLAS version 0.2.14.1
TEST(TestLinalg, TestComplexMatrixVectorProduct)
{
    // clang-format off
    std::vector<std::complex<double>> A_data = {
        {2.0, 3.0}, {4.0, 5.0}, {4.0, 5.0}, {6.0, 7.0}
    };
    std::vector<std::complex<double>> x_data = {
        {8.0, 7.0}, {5.0, 6.0}
    };
    std::vector<std::complex<double>> z_data = {
        {-15.0, 87.0}, {-15.0, 139.0}
    };
    // clang-format on
    Scilib::Matrix<std::complex<double>> A(A_data, 2, 2);
    Scilib::Vector<std::complex<double>> x(x_data, 2);
    Scilib::Vector<std::complex<double>> z(z_data, 2);

    EXPECT_EQ((A * x), z);
}
#endif
