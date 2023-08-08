// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/mdarray.h>
#include <scilib/statistics.h>
#include <vector>

TEST(TestStats, TestMean)
{
    Sci::Vector<double> v = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
                             40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0};
    EXPECT_NEAR(Sci::Stats::mean(v), 22.066666666666666, 1.0e-8);
}

TEST(TestStats, TestMedianOdd)
{
    Sci::Vector<double> v = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
                             40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0};
    EXPECT_EQ(Sci::Stats::median(v), 23.0);
}

TEST(TestStats, TestMedianEven)
{
    Sci::Vector<double> v = {3.0, 13.0, 7.0, 5.0, 21.0, 39.0, 40.0, 23.0, 14.0, 12.0, 56.0, 29.0};
    EXPECT_EQ(Sci::Stats::median(v), 17.5);
}

TEST(TestStats, TestStddev)
{
    Sci::Vector<double> v = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
                             40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0};
    EXPECT_NEAR(Sci::Stats::stddev(v), 14.49860420211283, 1.0e-8);
}

TEST(TestStats, TestRMS)
{
    Sci::Vector<double> v = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0, 23.0,
                             40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0};
    EXPECT_NEAR(Sci::Stats::rms(v), 26.136819495365792, 1.0e-8);
}

TEST(TestStats, TestCov)
{
    Sci::Vector<double> b = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 23.0,
                             40.0, 23.0, 14.0, 12.0, 56.0, 23.0, 29.0};
    Sci::Vector<double> c = {3.0,  13.0, 7.0,  5.0,  21.0, 23.0, 39.0,
                             23.0, 40.0, 23.0, 14.0, 12.0, 56.0, 23.0};
    EXPECT_NEAR(Sci::Stats::cov(b, c), 59.78021978, 1.0e-8);
}
