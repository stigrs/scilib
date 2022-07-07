// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <iostream>
#include <scilib/mdarray.h>
#include <vector>

int main()
{
    // clang-format off
    std::vector<int> data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };
    // clang-format on
    Sci::Matrix<int> m(data, 3, 4);
    std::cout << "matrix:\n" << m << '\n';

    Sci::Vector<int> r = Sci::row(m, 1);
    std::cout << "m.row(1):\n" << r << '\n';

    Sci::Vector<int> c = Sci::column(m, 2);
    std::cout << "m.column(2):\n" << c << '\n';

    auto sub = Sci::slice(m, Sci::seq(1, 3), Sci::seq(1, 4));
    std::cout << "m.sub:\n";
    Sci::print(std::cout, sub);
    std::cout << '\n';

    Sci::apply(sub, [&](int& i) { i *= -1; });
    std::cout << "matrix:\n" << m << '\n';
}
