// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestMatrixDecomposition, TestLU)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // clang-format off
    std::vector<double> a_data = {
        2.0, 5.0, 8.0, 7.0,
        5.0, 2.0, 2.0, 8.0,
        7.0, 5.0, 6.0, 6.0,
        5.0, 4.0, 4.0, 8.0 
    };
    std::vector<double> ans_data = {  
        7.000000,  5.00000,  6.000000, 6.00000,
        0.285714,  3.57143,  6.285710, 5.28571,
        0.714286,  0.12000, -1.040000, 3.08000,
        0.714286, -0.44000, -0.461538, 7.46154 
    };
    // clang-format on

    Matrix<double> ans(ans_data, 4, 4);
    Matrix<double> a(a_data, 4, 4);
    Vector<BLAS_INT> ipiv(4);
    lu(a.view(), ipiv.view());
    for (std::size_t i = 0; i < ans.rows(); ++i) {
        for (std::size_t j = 0; j < ans.cols(); ++j) {
            EXPECT_NEAR(a(i, j), ans(i, j), 1.0e-5);
        }
    }
}

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

TEST(TestMatrixDecomposition, TestSVD)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    // clang-format off
    std::vector<double> data = {
          8.79,  9.93,  9.83,  5.45,  3.16,   
          6.11,  6.91,  5.04, -0.27,  7.98,
         -9.15, -7.93,  4.86,  4.85,  3.01, 
          9.57,  1.64,  8.83,  0.74,  5.8,
         -3.49,  4.02,  9.80, 10.00,  4.27, 
          9.84,  0.15, -8.99, -6.02, -5.31};
    // clang-format on

    int m = 6;
    int n = 5;
    int ldu = m;
    int ldvt = n;

    Matrix<double> ans(data, m, n);
    Matrix<double> a(data, m, n);
    Vector<double> s(std::min(m, n));
    Matrix<double> u(m, ldu);
    Matrix<double> vt(n, ldvt);

    svd(a.view(), s.view(), u.view(), vt.view());

    Matrix<double> sigma(a.rows(), a.cols());
    auto sigma_diag = diag(sigma.view());
    sigma_diag = s;

    auto res = u * sigma * vt;
    for (std::size_t i = 0; i < ans.extent(0); ++i) {
        for (std::size_t j = 0; j < ans.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-6);
        }
    }
}