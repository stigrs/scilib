// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/diag.h>
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> v1_data = {10, 20, 30, 40};
    Scilib::Vector<int> v1(v1_data, v1_data.size());

    for (std::size_t i = 0; i < v1.extent(0); ++i) {
        std::cout << v1(i) << '\t';
    }
    std::cout << "\n\n";

    Scilib::Vector<int> v2(5);
    v2(0) = 1;
    std::cout << "v2(0) = " << v2(0) << '\n';

    // clang-format off
    std::vector<int> m1_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };
    // clang-format on
    Scilib::Matrix<int> m1(m1_data, 3, 4);

    std::cout << "rank() =    " << m1.rank() << '\n';
    std::cout << "extent(0) = " << m1.extent(0) << '\n';
    std::cout << "extent(1) = " << m1.extent(1) << '\n';
    for (std::size_t i = 0; i < m1.extent(0); ++i) {
        for (std::size_t j = 0; j < m1.extent(1); ++j) {
            std::cout << m1(i, j) << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    Scilib::Matrix<int> m2(m1.view());
    m2(0, 0) = 0;
    std::cout << "size() =    " << m2.size() << '\n';
    std::cout << "extent(0) = " << m2.extent(0) << '\n';
    std::cout << "extent(1) = " << m2.extent(1) << '\n';
    for (std::size_t i = 0; i < m2.extent(0); ++i) {
        for (std::size_t j = 0; j < m2.extent(1); ++j) {
            std::cout << m2(i, j) << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    m2.resize(4, 4);
    m2 = 5;
    std::cout << "size() =    " << m2.size() << '\n';
    std::cout << "extent(0) = " << m2.extent(0) << '\n';
    std::cout << "extent(1) = " << m2.extent(1) << '\n';
    for (std::size_t i = 0; i < m2.extent(0); ++i) {
        for (std::size_t j = 0; j < m2.extent(1); ++j) {
            std::cout << m2(i, j) << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    auto md = Scilib::diag(m2.view());
    md = 0;
    for (std::size_t i = 0; i < m2.extent(0); ++i) {
        for (std::size_t j = 0; j < m2.extent(1); ++j) {
            std::cout << m2(i, j) << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}
