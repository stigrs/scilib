// Copyright (c) 2023 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif

#include <iostream>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>

#if _MSC_VER
#pragma warning(pop)
#endif

int main()
{
    std::cout << Sci::Linalg::randn<Sci::Matrix<double>>(2, 3) << '\n';
}

