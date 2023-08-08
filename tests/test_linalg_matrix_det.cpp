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

TEST(TestLinalg, TestDet)
{
    using namespace Sci;
    using namespace Sci::Linalg;

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
    using extents_type = typename Matrix<double>::extents_type;

    Matrix<double> a2(extents_type(2, 2), a2_data);
    Matrix<double> a3(extents_type(3, 3), a3_data);
    Matrix<double> a4(extents_type(4, 4), a4_data);

    EXPECT_NEAR(det(a2), ans2, 1.0e-12);
    EXPECT_NEAR(det(a3), ans3, 1.0e-12);
    EXPECT_NEAR(det(a4), ans4, 1.0e-12);
}

TEST(TestLinalg, TestDetColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    const double ans4 = 242.0; // armadillo

    // clang-format off
    std::vector<double> a4_data = {
         1.0, -2.0,  5.0,  2.0,
         5.0,  3.0,  1.0,  3.0,
         4.0,  6.0,  0.0, -4.0,
         2.0,  4.0, -1.0,  0.0};
    // clang-format on
    using extents_type = typename Matrix<double, stdex::layout_left>::extents_type;

    Matrix<double, stdex::layout_left> a4(extents_type(4, 4), a4_data);
    EXPECT_NEAR(det(a4), ans4, 1.0e-12);
}
