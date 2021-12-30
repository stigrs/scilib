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

    EXPECT_NEAR(Scilib::Stats::mean(v), 22.066666666666666, 1.0e-8);
}

TEST(TestStats, TestMedianOdd)
{
    // clang-format off
    std::vector<double> data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
        40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0
    };
    // clang-format on
    Scilib::Vector<double> v(data, data.size());
    EXPECT_EQ(Scilib::Stats::median(v), 23.0);
}

TEST(TestStats, TestMedianEven)
{
    // clang-format off
    std::vector<double> data = {
        3.0,  13.0, 7.0,  5.0, 21.0, 39.0, 40.0, 23.0, 14.0, 12.0, 56.0, 29.0
    };
    // clang-format on
    Scilib::Vector<double> v(data, data.size());
    EXPECT_EQ(Scilib::Stats::median(v), 17.5);
}

TEST(TestStats, TestStddev)
{
    // clang-format off
    std::vector<double> data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
        40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0
    };
    // clang-format on
    Scilib::Vector<double> v(data, data.size());

    EXPECT_NEAR(Scilib::Stats::stddev(v), 14.49860420211283, 1.0e-8);
}

TEST(TestStats, TestRMS)
{
    // clang-format off
    std::vector<double> data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
        40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0
    };
    // clang-format on
    Scilib::Vector<double> v(data, data.size());

    EXPECT_NEAR(Scilib::Stats::rms(v), 26.136819495365792, 1.0e-8);
}

TEST(TestStats, TestCov)
{
    // clang-format off
    std::vector<double> b_data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 23.0,
        40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0
    };
    std::vector<double> c_data = {
        3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0,
        23.0, 40.0, 23.0, 14.0, 12.0, 56.0, 23.0
    };
    // clang-format on
    Scilib::Vector<double> b(b_data, b_data.size());
    Scilib::Vector<double> c(c_data, c_data.size());

    EXPECT_NEAR(Scilib::Stats::cov(b, c), 59.78021978, 1.0e-8);
}
