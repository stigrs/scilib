// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestVector, TestSize)
{
    std::size_t sz = 5;
    Sci::Vector<int> v(sz);
    EXPECT_EQ(v.size(), sz);
}

TEST(TestVector, TestElementAccesss)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
        EXPECT_EQ(v(i), i);
    }
}

TEST(TestVector, TestView)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }
    auto vv = v.view();
    EXPECT_EQ(vv(0), 0);
}

TEST(TestVector, TestCopy)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }

    Sci::Vector<int> a(v);
    a(0) = 10;
    EXPECT_EQ(v(0), 0);
    EXPECT_EQ(a(0), 10);
    EXPECT_NE(a(0), v(0));
}

TEST(TestVector, TestCopySpan)
{
    Sci::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }

    Sci::Vector<int> a(v.view());
    a(0) = 10;
    EXPECT_EQ(v(0), 0);
    EXPECT_EQ(a(0), 10);
    EXPECT_NE(a(0), v(0));
}

TEST(TestVector, TestCopyVector)
{
    std::array<int, 5> v{1, 1, 1, 1, 1};
    Sci::Vector<int> a(v, std::array<Sci::index, 1>{v.size()});

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i], a(i));
    }
    a *= 2;
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NE(v[i], a(i));
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

    std::swap(a, b);
    EXPECT_EQ(a.size(), n2);
    EXPECT_EQ(b.size(), n1);
}

TEST(TestVector, TestEmpty)
{
    Sci::Vector<int> a;
    EXPECT_TRUE(a.empty());
}

TEST(TestVector, TestInitializer)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    Sci::Vector<int> a(v, v.size());

    EXPECT_EQ(v.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(v[i], a(i));
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
        EXPECT_EQ(x(i), static_cast<int>(i + 1));
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
