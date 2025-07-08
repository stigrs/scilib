// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/mdarray.h>

void beast(const Sci::Array3D<int>& a)
{
    Sci::Array3D<int> b = Sci::slice(a, Kokkos::full_extent, Kokkos::full_extent, Kokkos::full_extent);
    b(1, 1, 1) = 666;
}

TEST(TestMDArray, TestArray3D)
{
    Sci::Array3D<int> a(3, 4, 5);
    a = 1;

    for (Sci::index i = 0; i < a.extent(0); ++i) {
        for (Sci::index j = 0; j < a.extent(1); ++j) {
            for (Sci::index k = 0; k < a.extent(2); ++k) {
                EXPECT_EQ(a(i, j, k), 1);
            }
        }
    }
}

TEST(TestMDArray, TestConstCorrectness)
{
    Sci::Array3D<int> a(5, 5, 5);
    Sci::Array3D<int> b(5, 5, 5);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 5; ++k) {
                a(i, j, k) = i + j + k;
                b(i, j, k) = i + j + k;
            }
        }
    }
    beast(a);
    EXPECT_EQ(a, b);
}

TEST(TestMDArray, TestArray3DInit)
{
    Sci::Array3D<int> m3 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};

    EXPECT_EQ(m3.rank(), 3);
    EXPECT_EQ(m3.size(), 12);
    EXPECT_EQ(m3.extent(0), 3);
    EXPECT_EQ(m3.extent(1), 2);
    EXPECT_EQ(m3.extent(2), 2);
}