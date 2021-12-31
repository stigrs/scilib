// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> v1_data = {10, 20, 30, 40};
    Sci::Vector<int> v1(v1_data, v1_data.size());
    std::cout << "v1 = " << v1 << "\n\n";

    Sci::Vector<int> v2(5);
    Sci::Linalg::fill(v2, 1);
    std::cout << "v2 = " << v2 << '\n';

    // clang-format off
    std::vector<int> m1_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };
    // clang-format on
    Sci::Matrix<int> m1(m1_data, 3, 4);
    std::cout << "m1.rank() =    " << m1.rank() << '\n';
    std::cout << "m1.extent(0) = " << m1.extent(0) << '\n';
    std::cout << "m1.extent(1) = " << m1.extent(1) << "\n\n";
    std::cout << "m1 = " << m1 << '\n';

    Sci::Matrix<int> m2(m1);
    m2(0, 0) = 0;
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = " << m2 << '\n';

    m2.resize(4, 4);
    m2 = 5;
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = " << m2 << "\n\n";

    Sci::Linalg::fill(Sci::diag(m2), 0);
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = " << m2 << "\n\n";

    using const_view_type = Sci::Matrix<int>::const_view_type;

    const_view_type m2_view = m2.view();
    std::cout << "m2_view.size() =    " << m2_view.size() << '\n';
    std::cout << "m2_view.extent(0) = " << m2_view.extent(0) << '\n';
    std::cout << "m2_view.extent(1) = " << m2_view.extent(1) << "\n\n";
    std::cout << "m2_view = ";
    Sci::print(std::cout, m2_view);
    std::cout << "\n\n";

    std::cout << "ssize(m2) =      " << Sci::ssize(m2) << '\n';
    std::cout << "sextent(m2, 0) = " << Sci::sextent(m2, 0) << '\n';
    std::cout << "sextent(m2, 1) = " << Sci::sextent(m2, 1) << "\n\n";

    // This should not compile:
    // m2_view(0, 0) = 1;
}
