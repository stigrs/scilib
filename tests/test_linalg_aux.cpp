// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalg, TestArgMaxArgMin)
{
    std::vector<int> data = {1, 2, 3, -4};
    Scilib::Vector<int> v(data, data.size());

    EXPECT_EQ(Scilib::Linalg::argmax(v), 2);
    EXPECT_EQ(Scilib::Linalg::argmin(v), 3);
}

TEST(TestLinalg, TestMaxMin)
{
    std::vector<int> data = {1, 2, 3, -4};
    Scilib::Vector<int> v(data, data.size());

    EXPECT_EQ(Scilib::Linalg::max(v), 3);
    EXPECT_EQ(Scilib::Linalg::min(v), -4);
}

TEST(TestLinalg, TestSumProd)
{
    std::vector<int> data = {1, 2, 3, 4};
    Scilib::Vector<int, stdex::layout_left> v(data, data.size());

    EXPECT_EQ(Scilib::Linalg::sum(v), 10);
    EXPECT_EQ(Scilib::Linalg::prod(v), 24);
}
