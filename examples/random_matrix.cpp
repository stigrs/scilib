// Copyright (c) 2023 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <iostream>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>

int main()
{
    std::cout << Sci::Linalg::randn<Sci::Matrix<double>>(2, 3) << '\n';
}

