// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <iostream>
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

    Sci::Vector<int> r = m.row(1);
    std::cout << "m.row(1):\n" << r << '\n';

    Sci::Vector<int> c = m.column(2);
    std::cout << "m.column(2):\n" << c << '\n';

    auto sub = stdex::submdspan(m.view(), std::pair<int, int>{1, 3},
                                std::pair<int, int>{1, 4});
    std::cout << "m.sub:\n";
    Sci::print(std::cout, sub);
    std::cout << '\n';

    for (std::size_t i = 0; i < sub.extent(0); ++i) {
        for (std::size_t j = 0; j < sub.extent(1); ++j) {
            sub(i, j) = 0;
        }
    }
    std::cout << "matrix:\n" << m << '\n';
}
