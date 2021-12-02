// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalg, TestElementWisePow)
{
    Scilib::Vector<double> ans(std::vector<double>{1, 4, 9, 16}, 4);
    Scilib::Vector<double> v(std::vector<double>{1, 2, 3, -4}, 4);

    auto res = Scilib::Linalg::pow(v, 2.0);
    for (std::size_t i = 0; i < ans.size(); ++i) {
        EXPECT_EQ(res(i), ans(i));
    }
}
