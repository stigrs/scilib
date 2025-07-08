// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <complex>
#include <gtest/gtest.h>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <vector>

TEST(TestLinalg, TestMatrixVectorProduct)
{
    std::vector<int> va = {1, -1, 2, 0, -3, 1};
    Sci::Matrix<int> a(Kokkos::dextents<Sci::index, 2>(2, 3), va);
    Sci::Vector<int> x(Kokkos::dextents<Sci::index, 1>(3), {2, 1, 0});
    Sci::Vector<int> y(Kokkos::dextents<Sci::index, 1>(2), {1, -3});
    EXPECT_EQ((a * x), y);
}

TEST(TestLinalg, TestMatrixVectorProductRowMajor)
{
    // clang-format off
    std::vector<double> a_data = {
         1.0, -1.0, 2.0,
         0.0, -3.0, 1.0};
    std::vector<double> x_data{2.0, 1.0, 0.0};
    std::vector<double> y_data{1.0, -3.0};
    // clang-format on
    Sci::Matrix<double> a(Kokkos::dextents<Sci::index, 2>(2, 3), a_data);
    Sci::Vector<double> x(Kokkos::dextents<Sci::index, 1>(x_data.size()), x_data);
    Sci::Vector<double> y(Kokkos::dextents<Sci::index, 1>(y_data.size()), y_data);
    Sci::Vector<double> res(y.size());

    Sci::Linalg::matrix_vector_product(a, x, res);
    EXPECT_EQ(res, y);
}

TEST(TestLinalg, TestMatrixVectorProductColMajor)
{
    // clang-format off
    std::vector<double> a_data = {
         1.0, 0.0,
        -1.0, -3.0,
         2.0, 1.0};
    std::vector<double> x_data{2.0, 1.0, 0.0};
    std::vector<double> y_data{1.0, -3.0};
    // clang-format on
    Sci::Matrix<double, Kokkos::layout_left> a(Kokkos::dextents<Sci::index, 2>(2, 3), a_data);
    Sci::Vector<double, Kokkos::layout_left> x(Kokkos::dextents<Sci::index, 1>(x_data.size()),
                                              x_data);
    Sci::Vector<double, Kokkos::layout_left> y(Kokkos::dextents<Sci::index, 1>(y_data.size()),
                                              y_data);
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
    Sci::Matrix<std::complex<double>> A(Kokkos::dextents<Sci::index, 2>(2, 2), A_data);
    Sci::Vector<std::complex<double>> x(Kokkos::dextents<Sci::index, 1>(x_data.size()), x_data);
    Sci::Vector<std::complex<double>> z(Kokkos::dextents<Sci::index, 1>(z_data.size()), z_data);

    EXPECT_EQ((A * x), z);
}
#endif
