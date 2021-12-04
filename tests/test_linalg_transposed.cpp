// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>

TEST(TestMatrix, TestTransposed)
{
    // clang-format off
    std::vector<int> data = {
        1,  2,  3,  4,  
        5,  6,  7,  8
    };
    std::vector<int> t_data = {
        1, 5,
        2, 6,
        3, 7,
        4, 8
    };
    // clang-format on
    Scilib::Matrix<int> a(data, 2, 4);
    Scilib::Matrix<int> ans(t_data, 4, 2);
    a = Scilib::Linalg::transposed(a.view());

    for (std::size_t i = 0; i < a.extent(0); ++i) {
        for (std::size_t j = 0; j < a.extent(1); ++j) {
            EXPECT_EQ(a(i, j), ans(i, j));
        }
    }
}
