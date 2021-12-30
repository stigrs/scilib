// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <iostream>
#include <vector>
#include <tuple>

int main()
{
    // clang-format off
    std::vector<int> data = {
        1,  2,  3, 
        4,  5,  6, 
        7,  8,  9
    };
    // clang-format on
    Scilib::Matrix<int> m(data, 3, 3);
    std::cout << "matrix:\n" << m << '\n';

    Scilib::Linalg::fill(Scilib::diag(m), 10);
    std::cout << "matrix:\n" << m << '\n';
}