// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <gtest/gtest.h>

TEST(TestMDArray, TestArray3D)
{
    Scilib::Array3D<int> a(3, 4, 5);
    a = 1;

    for (std::size_t i = 0; i < a.extent(0); ++i) {
        for (std::size_t j = 0; j < a.extent(1); ++j) {
            for (std::size_t k = 0; k < a.extent(2); ++k) {
                EXPECT_EQ(a(i, j, k), 1);
            }
        }
    }
}
