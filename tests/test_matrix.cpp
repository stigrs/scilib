// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <array>
#include <gtest/gtest.h>
#include <scilib/mdarray.h>

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

TEST(TestMatrix, TestElementAccessArray)
{
    Sci::Matrix<int> m(5, 3);
    m = 2;
    EXPECT_EQ(m[(std::array<Sci::index, 2>{0, 0})], 2);
    EXPECT_EQ(m[(std::array<Sci::index, 2>{4, 2})], 2);
}

TEST(TestMatrix, TestElementAccessSpan)
{
    Sci::Matrix<int> m(5, 3);
    m = 2;
    constexpr Sci::index a[] = {0, 0};
    constexpr Sci::index b[] = {4, 2};
    EXPECT_EQ(m[(std::span{a})], 2);
    EXPECT_EQ(m[(std::span{b})], 2);
}

TEST(TestMatrix, TestView)
{
    Sci::Matrix<int> m(5, 3);
    m = 2;
    auto mm = m.to_mdspan();
#if __cpp_multidimensional_subscript
    EXPECT_EQ(mm[0, 0], 2);
#else
    EXPECT_EQ(mm(0, 0), 2);
#endif
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
    Sci::Matrix<int> b(a.to_mdspan());
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
    Sci::Matrix<int> b = a.to_mdspan();
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

TEST(TestMatrix, TestSwapElements)
{
    std::array<int, 9> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 9> b = {10, 20, 30, 40, 50, 60, 70, 80, 90};

    Sci::StaticMatrix<int, 3, 3> aa(a);
    Sci::StaticMatrix<int, 3, 3> bb(b);

    Sci::swap_elements(aa.to_mdspan(), bb.to_mdspan());

    int it = 0;
    for (Sci::index i = 0; i < aa.extent(0); ++i) {
        for (Sci::index j = 0; j < aa.extent(1); ++j) {
            EXPECT_EQ(aa.at(i, j), b[it]);
            EXPECT_EQ(bb.at(i, j), a[it]);
            ++it;
        }
    }
}

TEST(TestMatrix, TestAeqB)
{
    std::array<int, 9> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 9> b = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Sci::StaticMatrix<int, 3, 3> aa(a);
    Sci::StaticMatrix<int, 3, 3> bb(b);

    EXPECT_TRUE(a == b);
}

TEST(TestMatrix, TestAnoteqB)
{
    std::array<int, 9> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 9> b = {10, 20, 30, 40, 50, 60, 70, 80, 90};

    Sci::StaticMatrix<int, 3, 3> aa(a);
    Sci::StaticMatrix<int, 3, 3> bb(b);

    EXPECT_TRUE(a != b);
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

TEST(TestMatrix, TestMDArrayInit)
{
    Sci::Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};

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
    Sci::Matrix<int> mm(m.mapping());
    mm = 3;
    for (Sci::index i = 0; i < mm.extent(0); ++i) {
        for (Sci::index j = 0; j < mm.extent(1); ++j) {
            EXPECT_EQ(mm(i, j), 3);
        }
    }
}

TEST(TestMatrix, TestAddValue)
{
    Sci::Matrix<int> m(5, 3);
    m = 1;
    m += 4;
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), 5);
        }
    }
}

TEST(TestMatrix, TestAddMatrix)
{
    Sci::Matrix<int> a(5, 3);
    a = 1;
    Sci::Matrix<int> b(5, 3);
    b = 4;
    Sci::Matrix<int> c = a + b;
    for (Sci::index i = 0; i < c.extent(0); ++i) {
        for (Sci::index j = 0; j < c.extent(1); ++j) {
            EXPECT_EQ(c(i, j), 5);
        }
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
    std::vector<int> ans = {1, 4, 2, 5, 3, 6};
    Sci::Matrix<int, stdex::layout_left> m = {{1, 2, 3}, {4, 5, 6}};

    int it = 0;
    for (Sci::index j = 0; j < m.extent(1); ++j) {
        for (Sci::index i = 0; i < m.extent(0); ++i) {
            EXPECT_EQ(m(i, j), ans[it]);
            ++it;
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
    Sci::Vector<int, stdex::layout_left> ans = {1, 5, 9};
    Sci::Matrix<int, stdex::layout_left> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto d = Sci::diag(m);

    int i = 0;
    for (auto it = Sci::begin(d); it != Sci::end(d); ++it) {
        EXPECT_EQ((*it), ans(i));
        ++i;
    }
}

TEST(TestMatrix, TestStaticMatrix33)
{
    Sci::StaticMatrix<int, 3, 3> m(3, 3);
    EXPECT_EQ(m.size(), 9);
    EXPECT_EQ(m.extent(0), 3);
    EXPECT_EQ(m.extent(1), 3);
    EXPECT_EQ(m.stride(0), 3);
    EXPECT_EQ(m.stride(1), 1);
    EXPECT_EQ(m.is_unique(), true);
    EXPECT_EQ(m.is_always_unique(), true);
    EXPECT_EQ(m.is_strided(), true);
    EXPECT_EQ(m.is_always_strided(), true);
    EXPECT_EQ(m.is_exhaustive(), true);
    EXPECT_EQ(m.is_always_exhaustive(), true);
}

TEST(TestMatrix, TestStaticMatrix53)
{
    Sci::StaticMatrix<int, 5, 3> m;
    EXPECT_EQ(m.size(), 5 * 3);
    EXPECT_EQ(m.extent(0), 5);
    EXPECT_EQ(m.extent(1), 3);
    EXPECT_EQ(m.stride(0), 3);
    EXPECT_EQ(m.stride(1), 1);
}

TEST(TestMatrix, TestStaticMatrixMdspan)
{
    Sci::Matrix<int> md = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    Sci::StaticMatrix<int, 2, 4> ms(md.to_mdspan());

    EXPECT_EQ(md.extent(0), ms.extent(0));
    EXPECT_EQ(md.extent(1), ms.extent(1));
}

TEST(TestMatrix, TestCopyRowMajorColMajor)
{
    Sci::Matrix<int> a = {{1, 2, 3}, {4, 5, 6}};
    Sci::Matrix<int, stdex::layout_left> b(a.to_mdspan());

    for (Sci::index j = 0; j < a.extent(1); ++j) {
        for (Sci::index i = 0; i < a.extent(0); ++i) {
            EXPECT_EQ(a(i, j), b(i, j));
        }
    }
}
