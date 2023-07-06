// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestCholeskyRowMajor)
{
    Sci::Matrix<double> A = {{4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98}};

    auto L = A;

    Sci::Linalg::cholesky(L);

    auto LT = Sci::Linalg::transposed(L);
    auto LLT = L * LT;

    for (Sci::index i = 0; i < A.extent(0); ++i) {
        for (Sci::index j = 0; j < A.extent(1); ++j) {
            EXPECT_NEAR(A(i, j), LLT(i, j), 1.0e-12);
        }
    }
}

TEST(TestLinalg, TestCholeskyColMajor)
{
    Sci::Matrix<double, stdex::layout_left> A = {
        {4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98}};

    auto L = A;

    Sci::Linalg::cholesky(L);

    auto LT = Sci::Linalg::transposed(L);
    auto LLT = L * LT;

    for (Sci::index i = 0; i < A.extent(0); ++i) {
        for (Sci::index j = 0; j < A.extent(1); ++j) {
            EXPECT_NEAR(A(i, j), LLT(i, j), 1.0e-12);
        }
    }
}

TEST(TestLinalg, TestLU)
{
    using namespace Sci;
    using namespace Sci::Linalg;

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

    lu(a, ipiv);

    for (Sci::index i = 0; i < ans.extent(0); ++i) {
        for (Sci::index j = 0; j < ans.extent(1); ++j) {
            EXPECT_NEAR(a(i, j), ans(i, j), 1.0e-5);
        }
    }
}

TEST(TestLinalg, TestQR)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> data = {
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0
    }; // clang-format on

    Matrix<double> ans(data, 3, 3);
    Matrix<double> a(data, 3, 3);
    Matrix<double> q(a.extent(0), a.extent(1));
    Matrix<double> r(a.extent(0), a.extent(1));

    qr(a, q, r);
    auto res = q * r;
    for (Sci::index i = 0; i < ans.extent(0); ++i) {
        for (Sci::index j = 0; j < ans.extent(1); ++j) {
            EXPECT_DOUBLE_EQ(res(i, j), ans(i, j));
        }
    }
}

TEST(TestLinalg, TestQRColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> data = {
         12.0,  6.0,  -4.0,
        -51.0, 167.0, 24.0,
          4.0, -68.0, -41.0
    }; // clang-format on

    Matrix<double, stdex::layout_left> ans(data, 3, 3);
    Matrix<double, stdex::layout_left> a(data, 3, 3);
    Matrix<double, stdex::layout_left> q(a.extent(0), a.extent(1));
    Matrix<double, stdex::layout_left> r(a.extent(0), a.extent(1));

    qr(a, q, r);
    auto res = q * r;
    for (Sci::index i = 0; i < ans.extent(0); ++i) {
        for (Sci::index j = 0; j < ans.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-8);
        }
    }
}

TEST(TestLinalg, TestSVD)
{
    using namespace Sci;
    using namespace Sci::Linalg;

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

    svd(a, s, u, vt);

    Matrix<double> sigma(a.extent(0), a.extent(1));
    auto sigma_diag = diag(sigma.view());
    copy_n(s.view(), std::min(m, n), sigma_diag);

    auto res = u * sigma * vt;
    for (Sci::index i = 0; i < ans.extent(0); ++i) {
        for (Sci::index j = 0; j < ans.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-6);
        }
    }
}

TEST(TestLinalg, TestSVDColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> data = {
          8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
          9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
          9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
          5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
          3.16,  7.98,  3.01,  5.8,   4.27, -5.31
    };
    // clang-format on

    int m = 6;
    int n = 5;
    int ldu = m;
    int ldvt = n;

    Matrix<double, stdex::layout_left> ans(data, m, n);
    Matrix<double, stdex::layout_left> a(data, m, n);
    Vector<double, stdex::layout_left> s(std::min(m, n));
    Matrix<double, stdex::layout_left> u(m, ldu);
    Matrix<double, stdex::layout_left> vt(n, ldvt);

    svd(a, s, u, vt);

    Matrix<double, stdex::layout_left> sigma(a.extent(0), a.extent(1));
    auto sigma_diag = diag(sigma.view());
    copy_n(s.view(), std::min(m, n), sigma_diag);

    auto res = u * sigma * vt;
    for (Sci::index j = 0; j < ans.extent(1); ++j) {
        for (Sci::index i = 0; i < ans.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans(i, j), 1.0e-6);
        }
    }
}
