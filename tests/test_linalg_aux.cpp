// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestArgMaxArgMin)
{
    Sci::Vector<int> v = {1, 2, 3, -4};

    EXPECT_EQ(Sci::Linalg::argmax(v), 2UL);
    EXPECT_EQ(Sci::Linalg::argmin(v), 3UL);
}

TEST(TestLinalg, TestMaxMin)
{
    Sci::Vector<int> v = {1, 2, 3, -4};

    EXPECT_EQ(Sci::Linalg::max(v), 3);
    EXPECT_EQ(Sci::Linalg::min(v), -4);
}

TEST(TestLinalg, TestSumProd)
{
    Sci::Vector<int, Kokkos::layout_left> v = {1, 2, 3, 4};

    EXPECT_EQ(Sci::Linalg::sum(v), 10);
    EXPECT_EQ(Sci::Linalg::prod(v), 24);
}

TEST(TestLinalg, TestZerosMatrix)
{
    auto m = Sci::Linalg::zeros<Sci::Matrix<int>>(2, 2);
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), 0);
        }
   }
}

TEST(TestLinalg, TestOnesMatrix)
{
    auto m = Sci::Linalg::ones<Sci::Matrix<int>>(2, 2);
    for (Sci::index i = 0; i < m.extent(0); ++i) {
        for (Sci::index j = 0; j < m.extent(1); ++j) {
            EXPECT_EQ(m(i, j), 1);
        }
    }
}