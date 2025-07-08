// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <vector>

TEST(TestLinalg, TestSolve)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> A_data = {
        1.0, 2.0, 3.0, 
        2.0, 3.0, 4.0, 
        3.0, 4.0, 1.0
    };
    std::vector<double> B_data = {
        14.0, 
        20.0, 
        14.0
    };
    std::vector<double> x = {1.0, 2.0, 3.0};
    // clang-format on

    using extents_type = typename Matrix<double>::extents_type;

    Matrix<double> A(extents_type(3, 3), A_data);
    Matrix<double> B(extents_type(3, 1), B_data);

    solve(A, B);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(B(i, 0), x[i], 1.0e-12);
    }
}

TEST(TestLinalg, TestSolveColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // clang-format off
    std::vector<double> A_data = {
        1.0, 2.0, 3.0, 
        2.0, 3.0, 4.0, 
        3.0, 4.0, 1.0
    };
    std::vector<double> B_data = {
        14.0, 
        20.0, 
        14.0
    };
    std::vector<double> x = {1.0, 2.0, 3.0};
    // clang-format on

    using extents_type = typename Matrix<double, Kokkos::layout_left>::extents_type;

    Matrix<double, Kokkos::layout_left> A(extents_type(3, 3), A_data);
    Matrix<double, Kokkos::layout_left> B(extents_type(3, 1), B_data);

    solve(A, B);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(B(i, 0), x[i], 1.0e-12);
    }
}
