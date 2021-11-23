// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalgBlas2, TestMatrixVectorProduct)
{
    std::vector<int> va = {1, -1, 2, 0, -3, 1};
    Sci::Matrix<int> a(va, 2, 3);
    Sci::Vector<int> x = {2, 1, 0};
    Sci::Vector<int> y = {1, -3};
    EXPECT_EQ((a * x), y);
}
