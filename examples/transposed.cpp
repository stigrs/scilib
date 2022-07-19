// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <iostream>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

int main()
{
    Sci::Matrix<int> m = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    std::cout << "matrix:\n" << m << "\n\n";

    auto mt = Sci::Linalg::transposed(m);
    std::cout << "transposed matrix:\n";
    Sci::print(std::cout, mt.view());
    std::cout << '\n';
}
