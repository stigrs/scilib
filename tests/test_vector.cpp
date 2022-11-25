// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <initializer_list>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestMDArray, TestSizeStaticVector)
{
    const int sz = 4;

    Sci::StaticVector<int, sz> v;

    EXPECT_EQ(v.extent(0), sz);
    EXPECT_EQ(v.size(), sz);
}

TEST(TestVector, TestEmpty)
{
    Sci::Vector<int> v;
    EXPECT_EQ(v.size(), 0);
}

TEST(TestVector, TestSize)
{
    std::size_t sz = 5;
    Sci::Vector<int> v(sz);
    EXPECT_EQ(v.size(), sz);
}

TEST(TestVector, TestAlloc)
{
    const std::size_t sz = 5;
    Sci::Vector<int> w(stdex::extents<int, sz>{sz}, std::allocator<int>());
    EXPECT_EQ(w.size(), sz);
}

TEST(TestVector, TestElementAccesss)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
        EXPECT_EQ(v[i], i);
    }
}

TEST(TestVector, TestView)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v[i] = i;
    }
    auto vv = v.view();
    EXPECT_EQ(vv[0], 0);
}

TEST(TestVector, TestCopy)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v[i] = i;
    }

    Sci::Vector<int> a(v);
    a(0) = 10;
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(a[0], 10);
    EXPECT_NE(a[0], v[0]);
}

TEST(TestVector, TestCopySpan)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v[i] = i;
    }

    Sci::Vector<int> a = Sci::make_mdarray(v.view());
    a(0) = 10;
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(a[0], 10);
    EXPECT_NE(a[0], v[0]);
}

TEST(TestVector, TestCopyVector)
{
    std::array<int, 5> v{1, 1, 1, 1, 1};
    Sci::StaticVector<int, 5> a(v, v.size());

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i], a[i]);
    }
    a *= 2;
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NE(v[i], a[i]);
    }
}

TEST(TestVector, TestResize)
{
    Sci::Vector<int> v(5);
    std::size_t sz = 10;
    v.resize(sz);
    EXPECT_EQ(v.size(), sz);
}

TEST(TestVector, TestSwap)
{
    std::size_t n1 = 5;
    std::size_t n2 = 10;

    Sci::Vector<int> a(n1);
    Sci::Vector<int> b(n2);

    a.swap(b);
    EXPECT_EQ(a.size(), n2);
    EXPECT_EQ(b.size(), n1);
}

TEST(TestVector, TestInitializer)
{
    std::initializer_list<int> v = {1, 2, 3, 4, 5};
    Sci::Vector<int> a(v, v.size());

    EXPECT_EQ(v.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(i + 1, a[i]);
    }
}

TEST(TestVector, TestMDArrayInit)
{
    Sci::Vector<int> a = {1, 2, 3, 4, 5};

    EXPECT_EQ(a.size(), 5);
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(i + 1, a.at(i));
    }
}

TEST(TestVector, TestSetValue)
{
    Sci::Vector<int> v(5);
    v = 4;
    for (const auto& vi : v) {
        EXPECT_EQ(vi, 4);
    }
}

TEST(TestVector, TestAddValue)
{
    Sci::Vector<int> v(5);
    v = 2;
    v += 4;
    for (const auto& vi : v) {
        EXPECT_EQ(vi, 6);
    }
}

TEST(TestVector, TestAddVector)
{
    Sci::Vector<int> a(5);
    a = 1;
    Sci::Vector<int> b(5);
    b = 1;
    a += b;
    for (const auto& ai : a) {
        EXPECT_EQ(ai, 2);
    }
    for (const auto& bi : b) {
        EXPECT_EQ(bi, 1);
    }
}

TEST(TestVector, TestAequalB)
{
    Sci::Vector<int> a(5);
    a = 1;
    Sci::Vector<int> b(5);
    b = 1;
    EXPECT_TRUE(a == b);
}

TEST(TestVector, TestAnotequalB)
{
    Sci::Vector<int> a(5);
    a = 1;
    Sci::Vector<int> b(4);
    b = 2;
    EXPECT_TRUE(a != b);
}

TEST(TestVector, TestSort)
{
    std::vector<int> data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    Sci::Vector<int> x(data, data.size());
    Sci::sort(x.view());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(x[i], static_cast<int>(i + 1));
    }
}

TEST(TestVector, TestFirst)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    Sci::Vector<int> a(v, v.size());

    auto v_slice = Sci::first(a.view(), v.size());
    for (std::size_t i = 0; i < v_slice.size(); ++i) {
        EXPECT_EQ(v_slice(i), a(i));
    }
}

TEST(TestVector, TestLast)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    Sci::Vector<int> a(v, v.size());

    auto v_slice = Sci::last(a.view(), 3);
    for (std::size_t i = 0; i < v_slice.size(); ++i) {
        EXPECT_EQ(v_slice(i), a(i + 2));
    }
}