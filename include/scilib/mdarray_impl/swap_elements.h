// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SWAP_ELEMENTS_H
#define SCILIB_MDARRAY_SWAP_ELEMENTS_H

#include <experimental/mdspan>
#include <utility>

namespace Scilib {

namespace stdex = std::experimental;

template <class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void
swap_elements(stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
              stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < x.extent(0); ++i) {
        std::swap(x(i), y(i));
    }
}

template <class T,
          stdex::extents<>::size_type nrows_x,
          stdex::extents<>::size_type ncols_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type nrows_y,
          stdex::extents<>::size_type ncols_y,
          class Layout_y,
          class Accessor_y>
inline void swap_elements(
    stdex::mdspan<T, stdex::extents<nrows_x, ncols_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T, stdex::extents<nrows_y, ncols_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < x.extent(0); ++i) {
        for (size_type j = 0; j < x.extent(1); ++j) {
            std::swap(x(i, j), y(i, j));
        }
    }
}

template <class T,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          class Layout_y,
          class Accessor_y>
inline void swap_elements(
    stdex::mdspan<T, stdex::extents<n1_x, n2_x, n3_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T, stdex::extents<n1_y, n2_y, n3_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < x.extent(0); ++i) {
        for (size_type j = 0; j < x.extent(1); ++j) {
            for (size_type k = 0; k < x.extent(2); ++k) {
                std::swap(x(i, j, k), y(i, j, k));
            }
        }
    }
}

template <class T,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          stdex::extents<>::size_type n4_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          stdex::extents<>::size_type n4_y,
          class Layout_y,
          class Accessor_y>
inline void swap_elements(
    stdex::
        mdspan<T, stdex::extents<n1_x, n2_x, n3_x, n4_x>, Layout_x, Accessor_x>
            x,
    stdex::
        mdspan<T, stdex::extents<n1_y, n2_y, n3_y, n4_y>, Layout_y, Accessor_y>
            y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));
    static_assert(x.static_extent(3) == y.static_extent(4));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < x.extent(0); ++i) {
        for (size_type j = 0; j < x.extent(1); ++j) {
            for (size_type k = 0; k < x.extent(2); ++k) {
                for (size_type l = 0; l < x.extent(3); ++l) {
                    std::swap(x(i, j, k, l), y(i, j, k, l));
                }
            }
        }
    }
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_SWAP_ELEMENTS_H
