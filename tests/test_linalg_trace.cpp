// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

TEST(TestMatrix, TestTrace)
{
    auto m_sym = Sci::Linalg::identity<Sci::Matrix<int>>(4);
    EXPECT_EQ(Sci::Linalg::trace(m_sym), 4);
}
