// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <gtest/gtest.h>

TEST(TestMatrix, TestSize)
{
    Sci::index nrows = 5;
    Sci::index ncols = 3;

    Sci::Matrix<int> m(nrows, ncols);

    EXPECT_EQ(m.size(), static_cast<std::size_t>(nrows * ncols));
    EXPECT_EQ(m.extent(0), nrows);
    EXPECT_EQ(m.extent(1), ncols);
}

TEST(TestMatrix, TestElementAccess)
{
    Sci::Matrix<int> m(5, 3);
    m = 2;
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), 2);
        }
    }
#ifndef NDEBUG
    EXPECT_DEATH(m(5, 2), "");
#endif
}

TEST(TestMatrix, TestView)
{
    Sci::Matrix<int> m(5, 3);
    m = 2;
    auto mm = m.view();
    EXPECT_EQ(mm(0, 0), 2);
}

TEST(TestMatrix, TestCopy)
{
    Sci::Matrix<int> a(5, 3);
    a = 2;
    Sci::Matrix<int> b(a);
    EXPECT_EQ(a(0, 0), b(0, 0));
}

TEST(TestMatrix, TestCopySpan)
{
    Sci::Matrix<int> a(5, 3);
    a = 2;
    Sci::Matrix<int> b(a.view());
    b(0, 0) = 3;
    EXPECT_EQ(a(0, 0), 2);
    EXPECT_EQ(b(0, 0), 3);
    EXPECT_NE(a(0, 0), b(0, 0));
}

TEST(TestMatrix, TestAssignSpan)
{
    Sci::Matrix<int> a(5, 3);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            a(i, j) = i + j;
        }
    }
    Sci::Matrix<int> b = a.view();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j), b(i, j));
        }
    }
}

TEST(TestMatrix, TestResize)
{
    Sci::index nrows = 5;
    Sci::index ncols = 3;
    Sci::Matrix<int> a(nrows, ncols);
    a.resize(3, 5);
    EXPECT_EQ(a.extent(0), ncols);
    EXPECT_EQ(a.extent(1), nrows);
}

TEST(TestMatrix, TestSwap)
{
    Sci::index n1 = 5;
    Sci::index n2 = 3;
    Sci::Matrix<int> a(n1, n2);
    Sci::Matrix<int> b(n2, n1);
    std::swap(a, b);
    EXPECT_EQ(a.extent(0), n2);
    EXPECT_EQ(a.extent(1), n1);
    EXPECT_EQ(b.extent(0), n1);
    EXPECT_EQ(b.extent(1), n2);
}

TEST(TestMatrix, TestInitializer)
{
    // clang-format off
    std::vector<int> v = {1, 2, 3,  
                          4, 5, 6};
    // clang-format on
    Sci::Matrix<int> m(v, 2, 3);

    int val = 1;
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), val);
            ++val;
        }
    }
}

TEST(TestMatrix, TestSetValue)
{
    Sci::Matrix<int> m(5, 3);
    m = 3;
    for (const auto& mi : m) {
        EXPECT_EQ(mi, 3);
    }
}

TEST(TestMatrix, TestAddValue)
{
    Sci::Matrix<int> m(5, 3);
    m = 1;
    m += 4;
    for (const auto& mi : m) {
        EXPECT_EQ(mi, 5);
    }
}

TEST(TestMatrix, TestAddMatrix)
{
    Sci::Matrix<int> a(5, 3);
    a = 1;
    Sci::Matrix<int> b(5, 3);
    b = 4;
    Sci::Matrix<int> c = a + b;
    for (const auto& ci : c) {
        EXPECT_EQ(ci, 5);
    }
}

TEST(TestMatrix, TestRow)
{
    // clang-format off
    std::vector<int> aa = {1, 2, 3, 
                           4, 5, 6};
    // clang-format on
    Sci::Matrix<int> ma(aa, 2, 3);

    auto r0 = Sci::row(ma, 0);
    auto r1 = Sci::row(ma, 1);

    for (std::size_t i = 0; i < r0.size(); ++i) {
        EXPECT_EQ(r0(i), static_cast<int>(i + 1));
    }
    for (std::size_t i = 0; i < r1.size(); ++i) {
        EXPECT_EQ(r1(i), static_cast<int>(i + 4));
    }
}

TEST(TestMatrix, TestDiag)
{
    // clang-format off
    std::vector<int> data = {
        1,  2,  3, 
        4,  5,  6, 
        7,  8,  9
    };
    std::vector<int> ans_data = {
        0,  2,  3, 
        4,  0,  6, 
        7,  8,  0
    };
    // clang-format on
    Sci::Matrix<int> ans(ans_data, 3, 3);
    Sci::Matrix<int> m(data, 3, 3);
    auto d = Sci::diag(m);
    Sci::apply(d, [&](int& i) { i = 0; });
    EXPECT_EQ(m, ans);
}

TEST(TestMatrix, TestColMajor)
{
    // clang-format off
    std::vector<int> v = {1, 4,  
                          2, 5,
                          3, 6};
    // clang-format on
    Sci::Matrix<int, stdex::layout_left> m(v, 2, 3);

    int val = 1;
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), val);
            ++val;
        }
    }
}

TEST(TestMatrix, TestRowIterator)
{
    // clang-format off
    std::vector<int> aa = {1, 2, 3, 
                           4, 5, 6};
    // clang-format on
    Sci::Matrix<int> ma(aa, 2, 3);

    auto r0 = Sci::row(ma, 0);

    int i = 1;
    for (auto it = Sci::cbegin(r0); it != Sci::cend(r0); ++it) {
        EXPECT_EQ((*it), i);
        ++i;
    }
}

TEST(TestMatrix, TestColIterator)
{
    // clang-format off
    std::vector<int> aa = {1, 2, 3, 
                           4, 5, 6};
    // clang-format on
    Sci::Matrix<int> ma(aa, 2, 3);

    auto c1 = Sci::column(ma, 1);

    int i = 2;
    for (auto it = Sci::cbegin(c1); it != Sci::cend(c1); ++it) {
        EXPECT_EQ((*it), i);
        i += 3;
    }
}

TEST(TestMatrix, TestDiagIterator)
{
    // clang-format off
    std::vector<int> data = {
        1,  2,  3, 
        4,  5,  6, 
        7,  8,  9
    };
    std::vector<int> ans_data = {
        1, 5, 9
    };
    // clang-format on
    Sci::Vector<int, stdex::layout_left> ans(ans_data, 3);
    Sci::Matrix<int, stdex::layout_left> m(data, 3, 3);

    auto d = Sci::diag(m);

    int i = 0;
    for (auto it = Sci::begin(d); it != Sci::end(d); ++it) {
        EXPECT_EQ((*it), ans(i));
        ++i;
    }
}
