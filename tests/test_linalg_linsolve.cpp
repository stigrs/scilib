// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestLinsolve)
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
    Matrix<double> A(A_data, 3, 3);
    Matrix<double> B(B_data, 3, 1);

    linsolve(A, B);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(B(i, 0), x[i], 1.0e-12);
    }
}

TEST(TestLinalg, TestLinsolveColMajor)
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
    Matrix<double, stdex::layout_left> A(A_data, 3, 3);
    Matrix<double, stdex::layout_left> B(B_data, 3, 1);

    linsolve(A, B);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(B(i, 0), x[i], 1.0e-12);
    }
}
