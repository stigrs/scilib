// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SORT_H
#define SCILIB_MDARRAY_SORT_H

#include <experimental/mdspan>
#include <algorithm>

namespace Scilib {
namespace stdex = std::experimental;

// Insertion sort.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline void sort(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    using size_type = stdex::extents<>::size_type;

    size_type i = 1;
    while (i < x.extent(0)) {
        size_type j = i;
        while (j > 0 && x(j - 1) > x(j)) {
            std::swap(x(j), x(j - 1));
            j -= 1;
        }
        i += 1;
    }
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_SORT_H
