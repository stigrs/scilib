// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <numlib/numlib.h>
#include <gtest/gtest.h>

TEST(TestVector, TestSize)
{
    Numlib::Vector<int> v(5);
    EXPECT_EQ(v.size(), 5);
}

TEST(TestVector, TestElementAccesss)
{
    Numlib::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
        EXPECT_EQ(v(i), i);
    }
}

TEST(TestVector, TestView)
{
    Numlib::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }
    auto vv = v.view();
    EXPECT_EQ(v(0), 0);
}

TEST(TestVector, TestCopy)
{
    Numlib::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }

    Numlib::Vector<int> a(v);
    a(0) = 10;
    EXPECT_EQ(v(0), 0);
    EXPECT_EQ(a(0), 10);
    EXPECT_NE(a(0), v(0));
}

TEST(TestVector, TestCopySpan)
{
    Numlib::Vector<int> v(5);
    for (int i = 0; i < 5; ++i) {
        v(i) = i;
    }

    Numlib::Vector<int> a(v.view());
    a(0) = 10;
    EXPECT_EQ(v(0), 0);
    EXPECT_EQ(a(0), 10);
    EXPECT_NE(a(0), v(0));
}

TEST(TestVector, TestCopyVector)
{
    std::vector<int> v(5, 1);
    Numlib::Vector<int> a = v;

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i], a(i));
    }
}

TEST(TestVector, TestResize)
{
    Numlib::Vector<int> v(5);
    v.resize(10);
    EXPECT_EQ(v.size(), 10);
}

TEST(TestVector, TestSwap)
{
    Numlib::Vector<int> a(5);
    Numlib::Vector<int> b(10);

    std::swap(a, b);
    EXPECT_EQ(a.size(), 10);
    EXPECT_EQ(b.size(), 5);
}

TEST(TestVector, TestEmpty)
{
    Numlib::Vector<int> a;
    EXPECT_TRUE(a.empty());
}

TEST(TestVector, TestInitializer)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    Numlib::Vector<int> a(v);

    EXPECT_EQ(v.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(v[i], a(i));
    }
}

TEST(TestVector, TestSetValue)
{
    Numlib::Vector<int> v(5, 2);
    v = 4;
    for (const auto& vi : v) {
        EXPECT_EQ(vi, 4);
    }
}

TEST(TestVector, TestAddValue)
{
    Numlib::Vector<int> v(5, 2);
    v += 4;
    for (const auto& vi : v) {
        EXPECT_EQ(vi, 6);
    }
}

TEST(TestVector, TestAddVector)
{
    Numlib::Vector<int> a(5, 1);
    Numlib::Vector<int> b(5, 1);
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
    Numlib::Vector<int> a(5, 1);
    Numlib::Vector<int> b(5, 1);
    EXPECT_TRUE(a == b);
}

TEST(TestVector, TestAnotequalB)
{
    Numlib::Vector<int> a(5, 1);
    Numlib::Vector<int> b(4, 2);
    EXPECT_TRUE(a != b);
}
