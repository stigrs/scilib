// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalg, TestMatrixVectorProduct)
{
    std::vector<int> va = {1, -1, 2, 0, -3, 1};
    Sci::Matrix<int> a(va, 2, 3);
    Sci::Vector<int> x(std::vector<int>{2, 1, 0}, 3);
    Sci::Vector<int> y(std::vector<int>{1, -3}, 2);
    EXPECT_EQ((a * x), y);
}
