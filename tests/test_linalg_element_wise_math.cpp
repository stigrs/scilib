// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestElementWisePow)
{
    Sci::Vector<double> ans(Mdspan::dextents<Sci::index, 1>(4), {1, 4, 9, 16});
    Sci::Vector<double> v(Mdspan::dextents<Sci::index, 1>(4), {1, 2, 3, -4});

    auto res = Sci::Linalg::pow(v, 2.0);
    for (std::size_t i = 0; i < ans.size(); ++i) {
        EXPECT_EQ(res(i), ans(i));
    }
}
