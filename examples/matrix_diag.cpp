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
    Sci::Matrix<int> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::cout << "matrix:\n" << m << '\n';

    Sci::Linalg::fill(Sci::diag(m), 10);
    std::cout << "matrix:\n" << m << '\n';
}