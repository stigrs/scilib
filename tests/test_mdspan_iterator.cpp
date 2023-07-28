// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <experimental/mdspan>
#include <gtest/gtest.h>
#include <scilib/mdarray.h>

TEST(TestMDSpanIterator, IteratorDefaultInit)
{
    Sci::MDSpan_iterator<int> it1;
    Sci::MDSpan_iterator<int> it2;
    EXPECT_TRUE(it1 == it2);
}

TEST(TestMDSpanIterator, IteratorComparisons)
{
    Sci::Vector<int> va = {1, 2, 3, 4};

    auto s = va.to_mdspan();

    auto it = Sci::begin(s);
    auto it2 = it + 1;

    EXPECT_TRUE(it == it);
    EXPECT_TRUE(it == Sci::begin(s));
    EXPECT_TRUE(Sci::begin(s) == it);

    EXPECT_TRUE(it != it2);
    EXPECT_TRUE(it2 != it);
    EXPECT_TRUE(it != Sci::end(s));
    EXPECT_TRUE(it2 != Sci::end(s));
    EXPECT_TRUE(Sci::end(s) != it);

    EXPECT_TRUE(it < it2);
    EXPECT_TRUE(it <= it2);
    EXPECT_TRUE(it2 <= Sci::end(s));
    EXPECT_TRUE(it < Sci::end(s));

    EXPECT_TRUE(it2 > it);
    EXPECT_TRUE(it2 >= it);
    EXPECT_TRUE(Sci::end(s) > it2);
    EXPECT_TRUE(Sci::end(s) >= it2);
}

TEST(TestMDSpanIterator, BeginEnd)
{
    Sci::Vector<int> va = {1, 2, 3, 4};

    auto s = va.to_mdspan();

    auto it = Sci::begin(s);
    auto first = it;
    EXPECT_TRUE(it == first);
    EXPECT_TRUE(*it == 1);

    auto beyond = Sci::end(s);
    EXPECT_TRUE(it != beyond);

    EXPECT_TRUE(beyond - first == 4);
    EXPECT_TRUE(first - first == 0);
    EXPECT_TRUE(beyond - beyond == 0);

    ++it;
    EXPECT_TRUE(it - first == 1);
    EXPECT_TRUE(*it == 2);
    *it = 22;
    EXPECT_TRUE(*it == 22);
    EXPECT_TRUE(beyond - it == 3);

    it = first;
    EXPECT_TRUE(it == first);
    while (it != Sci::end(s)) {
        *it = 5;
        ++it;
    }

    EXPECT_TRUE(it == beyond);
    EXPECT_TRUE(it - beyond == 0);

    for (auto iter = Sci::cbegin(s); iter != Sci::cend(s); ++iter) {
        EXPECT_TRUE((*iter) == 5);
    }
    for (auto iter = Sci::end(s); iter-- != Sci::begin(s);) {
        EXPECT_TRUE((*iter) == 5);
    }
}
