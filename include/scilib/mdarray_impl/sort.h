// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SORT_H
#define SCILIB_MDARRAY_SORT_H

#include <algorithm>

namespace Sci {

namespace __Detail {

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline std::size_t partition(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x,
                             std::size_t start,
                             std::size_t end)
{
    T xi = x[start];
    std::size_t i = start;
    for (std::size_t j = start + 1; j < end; ++j) {
        if (x[j] <= xi) {
            ++i;
            std::swap(x[i], x[j]);
        }
    }
    std::swap(x[i], x[start]);
    return i;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void quick_sort(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x,
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
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void sort(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    using index_type = IndexType;

    index_type start = 0;
    index_type end = x.extent(0);
    __Detail::quick_sort(x, start, end);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SORT_H
