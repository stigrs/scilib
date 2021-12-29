// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>

TEST(TestMatrix, TestTrace)
{
    auto m_sym = Scilib::Linalg::identity<Scilib::Matrix<int>>(4);
    EXPECT_EQ(Scilib::Linalg::trace(m_sym.view()), 4);
}
