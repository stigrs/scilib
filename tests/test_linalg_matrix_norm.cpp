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

TEST(TestLinalg, TestMatrixNorm)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> a_data = {
        -3.0, 5.0, 7.0,
         0.0, 2.0, 8.0
    };
    // clang-format on
    using extents_type = typename Matrix<double>::extents_type;
    Matrix<double> A(extents_type(2, 3), a_data);

    EXPECT_EQ(matrix_norm(A, '1'), 15.0);
    EXPECT_EQ(matrix_norm(A, 'I'), 15.0);
}

TEST(TestLinalg, TestMatrixNormColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> a_data = {
        -3.0, 0.0,
         5.0, 2.0,
         7.0, 8.0
    };
    // clang-format on
    using extents_type = typename Matrix<double, Mdspan::layout_left>::extents_type;
    Matrix<double, Mdspan::layout_left> A(extents_type(2, 3), a_data);

    EXPECT_EQ(matrix_norm(A, '1'), 15.0);
    EXPECT_EQ(matrix_norm(A, 'I'), 15.0);
}

TEST(TestLinalg, TestMatrixNormComplex)
{
    Sci::Matrix<std::complex<double>> A = {{{1.0, 0.0}, {0.0, -2.0}}, {{0.0, 2.0}, {5.0, 0.0}}};
    EXPECT_NEAR(Sci::Linalg::matrix_norm(A, 'F'), 5.830951894845301, 1.0e-9);
}