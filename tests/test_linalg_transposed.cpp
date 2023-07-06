// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <array>
#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>


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
    Sci::Matrix<int> a(data, 2, 4);
    Sci::Matrix<int> ans(t_data, 4, 2);
    auto at = Sci::Linalg::transposed(a);

    for (Sci::index i = 0; i < at.extent(0); ++i) {
        for (Sci::index j = 0; j < at.extent(1); ++j) {
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
    Sci::Matrix<int, stdex::layout_left> a(data, 4, 2);
    Sci::Matrix<int, stdex::layout_left> ans(t_data, 2, 4);
    auto at = Sci::Linalg::transposed(a);

    for (Sci::index j = 0; j < at.extent(1); ++j) {
        for (Sci::index i = 0; i < at.extent(0); ++i) {
            EXPECT_EQ(at(i, j), ans(i, j));
        }
    }
}

TEST(TestMatrix, TestTransposedStatic)
{
    // clang-format off
    std::array<int, 8> data = {
        1,  2,  3,  4,  
        5,  6,  7,  8
    };
    std::array<int, 8> t_data = {
        1, 5,
        2, 6,
        3, 7,
        4, 8
    };
    // clang-format on
    Sci::StaticMatrix<int, 2, 4> a(data, 2, 4);
    Sci::StaticMatrix<int, 4, 2> ans(t_data, 4, 2);
    auto at = Sci::Linalg::transposed(a);

    for (Sci::index i = 0; i < at.extent(0); ++i) {
        for (Sci::index j = 0; j < at.extent(1); ++j) {
            EXPECT_EQ(at(i, j), ans(i, j));
        }
    }
}