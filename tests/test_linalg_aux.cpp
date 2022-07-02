// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestArgMaxArgMin)
{
    std::vector<int> data = {1, 2, 3, -4};
    Sci::Vector<int> v(data, data.size());

    EXPECT_EQ(Sci::Linalg::argmax(v), 2UL);
    EXPECT_EQ(Sci::Linalg::argmin(v), 3UL);
}

TEST(TestLinalg, TestMaxMin)
{
    std::vector<int> data = {1, 2, 3, -4};
    Sci::Vector<int> v(data, data.size());

    EXPECT_EQ(Sci::Linalg::max(v), 3);
    EXPECT_EQ(Sci::Linalg::min(v), -4);
}

TEST(TestLinalg, TestSumProd)
{
    std::vector<int> data = {1, 2, 3, 4};
    Sci::Vector<int, stdex::layout_left> v(data, data.size());

    EXPECT_EQ(Sci::Linalg::sum(v), 10);
    EXPECT_EQ(Sci::Linalg::prod(v), 24);
}
