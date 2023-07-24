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
    Sci::Matrix<int, stdex::layout_left> b(2, 3);

    auto multiply = [&](int i, int j) {
        m(i, j) *= 2;
    };
    auto copy = [&](int i, int j) {
        b(i, j) = m(i, j);
    };
    auto printer = [&](int i, int j) {
        std::cout << "b(" << i << "," << j << ") = " << b(i, j) << "\n";
    };
    Sci::for_each_in_extents(multiply, m);
    Sci::for_each_in_extents(copy, m);
    Sci::for_each_in_extents(printer, b);
}