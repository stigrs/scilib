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
    auto at = Scilib::Linalg::transposed(a);

    for (std::size_t i = 0; i < at.extent(0); ++i) {
        for (std::size_t j = 0; j < at.extent(1); ++j) {
            EXPECT_EQ(at(i, j), ans(i, j));
        }
    }
}

TEST(TestMatrix, TestTransposedColMajor)
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
    Scilib::Matrix<int, stdex::layout_left> a(data, 4, 2);
    Scilib::Matrix<int, stdex::layout_left> ans(t_data, 2, 4);
    auto at = Scilib::Linalg::transposed(a);

    for (std::size_t j = 0; j < at.extent(1); ++j) {
        for (std::size_t i = 0; i < at.extent(0); ++i) {
            EXPECT_EQ(at(i, j), ans(i, j));
        }
    }
}
