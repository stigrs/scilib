// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SORT_H
#define SCILIB_MDARRAY_SORT_H

#include <algorithm>

namespace Sci {
namespace stdex = std::experimental;

namespace __Detail {

template <class T, std::size_t ext, class Layout, class Accessor>
inline std::size_t partition(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> x,
                             std::size_t start,
                             std::size_t end)
{
    using size_type = std::size_t;

    T xi = x(start);
    size_type i = start;
    for (size_type j = start + 1; j < end; ++j) {
        if (x(j) <= xi) {
            ++i;
            std::swap(x(i), x(j));
        }
    }
    std::swap(x(i), x(start));
    return i;
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline void quick_sort(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> x,
                       std::size_t start,
                       std::size_t end)
{
    if (start < end) {
        auto pivot = partition(x, start, end);
        quick_sort(x, start, pivot);
        quick_sort(x, pivot + 1, end);
    }
}

} // namespace __Detail

// Quick sort.
template <class T, std::size_t ext, class Layout, class Accessor>
inline void sort(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> x)
{
    using size_type = std::size_t;

    size_type start = 0;
    size_type end = x.extent(0);
    __Detail::quick_sort(x, start, end);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SORT_H
