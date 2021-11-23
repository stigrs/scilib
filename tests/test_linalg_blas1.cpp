// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalgBlas1, TestAxpy)
{
    std::vector<int> ans = {4, 8, 12, 16, 20};
    Sci::Vector<int> x = {1, 2, 3, 4, 5};
    Sci::Vector<int> y = {2, 4, 6, 8, 10};

    Sci::axpy(2, x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(y(i), ans[i]);
    }
}

TEST(TestLinalgBlas1, TestDot)
{
    Sci::Vector<int> a = {1, 3, -5};
    Sci::Vector<int> b = {4, -2, -1};

    EXPECT_EQ(Sci::dot_product(a, b), 3);
}
