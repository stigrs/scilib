// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <experimental/linalg>
#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinAlg, TestAdd)
{
    std::vector<int> data = {1, 2, 3};
    std::vector<int> ans = {2, 4, 6};

    Sci::Vector<int> x(data, data.size());
    Sci::Vector<int> y(data, data.size());
    Sci::Vector<int> z(data.size());

    Sci::Linalg::add(x, y, z);

    for (std::size_t i = 0; i < z.size(); ++i) {
        EXPECT_EQ(z(i), ans[i]);
    }
}

TEST(TestLinAlg, TestSubtract)
{
    std::vector<int> ans = {1, 2, 3};

    Sci::Vector<int> x = {1, 2, 3};
    Sci::Vector<int> y = {2, 4, 6};
    Sci::Vector<int> z = y - x;

    for (std::size_t i = 0; i < z.size(); ++i) {
        EXPECT_EQ(z(i), ans[i]);
    }
}

TEST(TestLinalg, TestAbsSum)
{
    Sci::Vector<int> v({1, 2, 3, -4}, 4);
    EXPECT_EQ(Sci::Linalg::vector_abs_sum(v), 10);
}

TEST(TestLinalg, TestAxpy)
{
    std::vector<int> ans = {4, 8, 12, 16, 20};
    Sci::Vector<int> x({1, 2, 3, 4, 5}, 5);
    Sci::Vector<int> y({2, 4, 6, 8, 10}, 5);

    Sci::Linalg::axpy(2, x, y);
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_EQ(y(i), ans[i]);
    }
}

TEST(TestLinalg, TestDot)
{
    Sci::Vector<int> a({1, 3, -5}, 3);
    Sci::Vector<int> b({4, -2, -1}, 3);

    EXPECT_EQ(Sci::Linalg::dot(a, b), 3);
}

TEST(TestLinalg, TestIdxAbsMax)
{
    Sci::Vector<int> v({1, 3, -5, 2}, 4);
    EXPECT_EQ(Sci::Linalg::idx_abs_max(v), 2UL);
}

TEST(TestLinalg, TestIdxAbsMin)
{
    Sci::Vector<int> v({4, 3, -5, 2}, 4);
    EXPECT_EQ(Sci::Linalg::idx_abs_min(v), 3UL);
}

TEST(TestLinalg, TestNorm2)
{
    Sci::Vector<double> v({1.0, 2.0, 3.0}, 3);
    auto ans = Sci::Linalg::vector_norm2(v);
    EXPECT_NEAR(ans * ans, 14.0, 1.0e-12);
}

TEST(TestLinalg, TestNorm2Row)
{
    // clang-format off
    std::vector<double> aa = {1, 2, 3, 
                              4, 5, 6};
    // clang-format on
    Sci::Matrix<double> ma(aa, 2, 3);
    auto ans = std::experimental::linalg::vector_norm2(Sci::row(ma.to_mdspan(), 0));
    EXPECT_NEAR(ans * ans, 14.0, 1.0e-12);
}

TEST(TestLinalg, TestScaled)
{
    // clang-format off
    std::vector<double> a_data = {1, 2, 3, 4, 5, 6};
    // clang-format on
    Sci::Vector<double> a(a_data, 6);
    auto ans = Sci::Linalg::scaled(2.0, a);

    for (std::size_t i = 0; i < ans.size(); ++i) {
        EXPECT_EQ(ans(i), 2.0 * (1.0 + i));
    }
}
