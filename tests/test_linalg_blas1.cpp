// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalgBlas1, TestAbsSum)
{
    Sci::Vector<int> v = {1, 2, 3, -4};
    EXPECT_EQ(Sci::Linalg::abs_sum(v.view()), 10);
}

TEST(TestLinalgBlas1, TestAxpy)
{
    std::vector<int> ans = {4, 8, 12, 16, 20};
    Sci::Vector<int> x = {1, 2, 3, 4, 5};
    Sci::Vector<int> y = {2, 4, 6, 8, 10};

    Sci::Linalg::axpy(2, x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(y(i), ans[i]);
    }
}

TEST(TestLinalgBlas1, TestDot)
{
    Sci::Vector<int> a = {1, 3, -5};
    Sci::Vector<int> b = {4, -2, -1};

    EXPECT_EQ(Sci::Linalg::dot_product(a, b), 3);
}

TEST(TestLinalgBlas1, TestIdxAbsMax)
{
    Sci::Vector<int> v = {1, 3, -5, 2};
    EXPECT_EQ(Sci::Linalg::idx_abs_max(v.view()), 2);
}

TEST(TestLinalgBlas1, TestIdxAbsMin)
{
    Sci::Vector<int> v = {1, 3, -5, 2};
    EXPECT_EQ(Sci::Linalg::idx_abs_min(v.view()), 0);
}

TEST(TestLinalgBlas1, TestNorm2)
{
    Sci::Vector<double> v = {1.0, 2.0, 3.0};
    auto ans = Sci::Linalg::norm2(v.view());
    EXPECT_EQ(ans * ans, 14.0);
}

TEST(TestMatrix, TestNorm2Row)
{
    // clang-format off
    std::vector<double> aa = {1, 2, 3, 
                              4, 5, 6};
    // clang-format on
    Sci::Matrix<double> ma(aa, 2, 3);
    auto ans = Sci::Linalg::norm2(ma.row(0));
    EXPECT_EQ(ans * ans, 14.0);
}