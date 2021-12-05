// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/statistics.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestStats, TestMean)
{
    // clang-format off
    std::vector<double> data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
        40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0
    };
    // clang-format on
    Scilib::Vector<double> v(data, data.size());

    EXPECT_NEAR(Scilib::Stats::mean(v.view()), 22.066666666666666, 1.0e-8);
}
