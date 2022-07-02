// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/mdarray.h>

TEST(TestMDArray, TestArray4D)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 1;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; j < a.extent(1); ++j) {
            for (index_type k = 0; k < a.extent(2); ++k) {
                for (index_type l = 0; l < a.extent(3); ++l) {
                    EXPECT_EQ(a(i, j, k, l), 1);
                }
            }
        }
    }
}
