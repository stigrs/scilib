// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <iostream>
#include <scilib/mdarray.h>

int main()
{
    Sci::Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
    Sci::Matrix<int, Mdspan::layout_left> b(2, 3);

    auto multiply = [&](Sci::index i, Sci::index j) { m(i, j) *= 2; };
    auto copy = [&](Sci::index i, Sci::index j) { b(i, j) = m(i, j); };
    auto printer = [&](Sci::index i, Sci::index j) {
        std::cout << "b(" << i << "," << j << ") = " << b(i, j) << "\n";
    };
    Sci::for_each_in_extents(multiply, m);
    Sci::for_each_in_extents(copy, m);
    Sci::for_each_in_extents(printer, b);
}