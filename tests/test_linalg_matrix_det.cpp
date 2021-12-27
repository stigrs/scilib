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

TEST(TestLinalg, TestDet)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    const double ans2 = 13.0;
    const double ans3 = 76.0;
    const double ans4 = 242.0; // armadillo

    // clang-format off
    std::vector<double> a2_data = {
         1.0, 5.0, 
        -2.0, 3.0
    };
    std::vector<double> a3_data = {
         1.0, 5.0, 4.0, 
        -2.0, 3.0, 6.0, 
         5.0, 1.0, 0.0
    };
    std::vector<double> a4_data = {
         1.0, 5.0,  4.0,  2.0,
        -2.0, 3.0,  6.0,  4.0,
         5.0, 1.0,  0.0, -1.0,
         2.0, 3.0, -4.0,  0.0
    };
    // clang-format on

    Matrix<double> a2(a2_data, 2, 2);
    Matrix<double> a3(a3_data, 3, 3);
    Matrix<double> a4(a4_data, 4, 4);

    EXPECT_NEAR(det(a2.view()), ans2, 1.0e-12);
    EXPECT_NEAR(det(a3.view()), ans3, 1.0e-12);
    EXPECT_NEAR(det(a4.view()), ans4, 1.0e-12);
}

TEST(TestLinalg, TestDetColMajor)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    const double ans4 = 242.0; // armadillo

    // clang-format off
    std::vector<double> a4_data = {
         1.0, -2.0,  5.0,  2.0,
         5.0,  3.0,  1.0,  3.0,
         4.0,  6.0,  0.0, -4.0,
         2.0,  4.0, -1.0,  0.0};
    // clang-format on

    Matrix<double, stdex::layout_left> a4(a4_data, 4, 4);
    EXPECT_NEAR(det(a4.view()), ans4, 1.0e-12);
}
