// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <gtest/gtest.h>

void beast(const Sci::Array3D<int>& a)
{
    Sci::Array3D<int> b = Sci::slice(a, stdex::full_extent, stdex::full_extent, stdex::full_extent);
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
