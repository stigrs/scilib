// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <numlib/numlib.h>
#include <gtest/gtest.h>

TEST(TestMatrix, TestSize)
{
    Numlib::Matrix<int> m(5, 3);
    EXPECT_EQ(m.size(), 5 * 3);
    EXPECT_EQ(m.rows(), 5);
    EXPECT_EQ(m.cols(), 3);
}

TEST(TestMatrix, TestElementAccess)
{
    Numlib::Matrix<int> m(5, 3, 2);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            EXPECT_EQ(m(i, j), 2);
        }
    }
}

TEST(TestMatrix, TestView)
{
    Numlib::Matrix<int> m(5, 3, 2);
    auto mm = m.view();
    EXPECT_EQ(mm(0, 0), 2);
}

TEST(TestMatrix, TestCopy)
{
    Numlib::Matrix<int> a(5, 3, 2);
    Numlib::Matrix<int> b(a);
    EXPECT_EQ(a(0, 0), b(0, 0));
}

TEST(TestMatrix, TestCopySpan)
{
    Numlib::Matrix<int> a(5, 3, 2);
    Numlib::Matrix<int> b(a.view());
    b(0, 0) = 3;
    EXPECT_EQ(a(0, 0), 2);
    EXPECT_EQ(b(0, 0), 3);
    EXPECT_NE(a(0, 0), b(0, 0));
}

TEST(TestMatrix, TestAssignSpan)
{
    Numlib::Matrix<int> a(5, 3);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            a(i, j) = i + j;
        }
    }
    Numlib::Matrix<int> b = a.view();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j), b(i, j));
        }
    }
}

TEST(TestMatrix, TestResize)
{
    Numlib::Matrix<int> a(5, 3);
    a.resize(3, 5);
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(a.cols(), 5);
}

TEST(TestMatrix, TestSwap)
{
    Numlib::Matrix<int> a(5, 3);
    Numlib::Matrix<int> b(3, 5);
    std::swap(a, b);
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(a.cols(), 5);
    EXPECT_EQ(b.rows(), 5);
    EXPECT_EQ(b.cols(), 3);
}

TEST(TestMatrix, TestInitializer)
{
    // clang-format off
    std::vector<int> v = {1, 2, 3,  
                          4, 5, 6};
    // clang-format on
    Numlib::Matrix<int> m(v, 2, 3);

    int val = 1;
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            EXPECT_EQ(m(i, j), val);
            ++val;
        }
    }
}

TEST(TestMatrix, TestSetValue)
{
    Numlib::Matrix<int> m(5, 3);
    m = 3;
    for (const auto& mi : m) {
        EXPECT_EQ(mi, 3);
    }
}

TEST(TestMatrix, TestAddValue)
{
    Numlib::Matrix<int> m(5, 3, 1);
    m += 4;
    for (const auto& mi : m) {
        EXPECT_EQ(mi, 5);
    }
}

TEST(TestMatrix, TestAddMatrix)
{
    Numlib::Matrix<int> a(5, 3, 1);
    Numlib::Matrix<int> b(5, 3, 4);
    a += b;
    for (const auto& ai : a) {
        EXPECT_EQ(ai, 5);
    }
}