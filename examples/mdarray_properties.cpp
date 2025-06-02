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
    Sci::Vector<int> v1 = {10, 20, 30, 40};
    std::cout << "v1 = \n" << v1 << "\n\n";

    Sci::Vector<int> v2(5);
    Sci::Linalg::fill(v2, 1);
    std::cout << "v2 = \n" << v2 << '\n';

    Sci::Matrix<int> m1 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    std::cout << "m1.rank() =    " << m1.rank() << '\n';
    std::cout << "m1.extent(0) = " << m1.extent(0) << '\n';
    std::cout << "m1.extent(1) = " << m1.extent(1) << "\n\n";
    std::cout << "m1 = \n" << m1 << '\n';

    Sci::Matrix<int> m2(m1);
    m2(0, 0) = 0;
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = \n" << m2 << '\n';

    m2.resize(4, 4);
    m2 = 5;
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = \n" << m2 << "\n\n";

    Sci::Linalg::fill(Sci::diag(m2), 0);
    std::cout << "m2.size() =    " << m2.size() << '\n';
    std::cout << "m2.extent(0) = " << m2.extent(0) << '\n';
    std::cout << "m2.extent(1) = " << m2.extent(1) << "\n\n";
    std::cout << "m2 = \n" << m2 << "\n\n";

    using const_mdspan_type = Sci::Matrix<int>::const_mdspan_type;

    const_mdspan_type m2_view = m2.to_mdspan();
    std::cout << "m2_view.size() =    " << m2_view.size() << '\n';
    std::cout << "m2_view.extent(0) = " << m2_view.extent(0) << '\n';
    std::cout << "m2_view.extent(1) = " << m2_view.extent(1) << "\n\n";
    std::cout << "m2_view = \n";
    Sci::print(std::cout, m2_view);
    std::cout << "\n\n";

    // This should not compile:
    // m2_view(0, 0) = 1;

    Sci::StaticMatrix<double, 2, 3> sv(Mdspan::extents<Sci::index, 2, 3>(), 2.0);
    std::cout << "sv = \n" << sv << '\n';
}
