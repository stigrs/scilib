// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinAlg, TestAdd)
{
    std::vector<int> data = {1, 2, 3};
    std::vector<int> ans = {2, 4, 6};

    Scilib::Vector<int> x(data, data.size());
    Scilib::Vector<int> y(data, data.size());
    Scilib::Vector<int> z(data.size());

    Scilib::Linalg::add(x, y, z);

    for (std::size_t i = 0; i < z.size(); ++i) {
        EXPECT_EQ(z(i), ans[i]);
    }
}

TEST(TestLinalg, TestAbsSum)
{
    Scilib::Vector<int> v(std::vector<int>{1, 2, 3, -4}, 4);
    EXPECT_EQ(Scilib::Linalg::abs_sum(v), 10);
}

TEST(TestLinalg, TestAxpy)
{
    std::vector<int> ans = {4, 8, 12, 16, 20};
    Scilib::Vector<int> x(std::vector<int>{1, 2, 3, 4, 5}, 5);
    Scilib::Vector<int> y(std::vector<int>{2, 4, 6, 8, 10}, 5);

    Scilib::Linalg::axpy(2, x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(y(i), ans[i]);
    }
}

TEST(TestLinalg, TestDot)
{
    Scilib::Vector<int> a(std::vector<int>{1, 3, -5}, 3);
    Scilib::Vector<int> b(std::vector<int>{4, -2, -1}, 3);

    EXPECT_EQ(Scilib::Linalg::dot(a, b), 3);
}

TEST(TestLinalg, TestIdxAbsMax)
{
    Scilib::Vector<int> v(std::vector<int>{1, 3, -5, 2}, 4);
    EXPECT_EQ(Scilib::Linalg::idx_abs_max(v), 2UL);
}

TEST(TestLinalg, TestIdxAbsMin)
{
    Scilib::Vector<int> v(std::vector<int>{1, 3, -5, 2}, 4);
    EXPECT_EQ(Scilib::Linalg::idx_abs_min(v), 0UL);
}

TEST(TestLinalg, TestNorm2)
{
    Scilib::Vector<double> v(std::vector<double>{1.0, 2.0, 3.0}, 3);
    auto ans = Scilib::Linalg::norm2(v);
    EXPECT_EQ(ans * ans, 14.0);
}

TEST(TestLinalg, TestNorm2Row)
{
    // clang-format off
    std::vector<double> aa = {1, 2, 3, 
                              4, 5, 6};
    // clang-format on
    Scilib::Matrix<double> ma(aa, 2, 3);
    auto ans = Scilib::Linalg::norm2(Scilib::row(ma.view(), 0));
    EXPECT_EQ(ans * ans, 14.0);
}

TEST(TestLinalg, TestScaled)
{
    // clang-format off
    std::vector<double> a_data = {1, 2, 3, 4, 5, 6};
    // clang-format on
    Scilib::Vector<double> a(a_data, 6);
    auto ans = Scilib::Linalg::scaled(2.0, a);

    for (std::size_t i = 0; i < ans.size(); ++i) {
        EXPECT_EQ(ans(i), 2.0 * (1.0 + i));
    }
}
