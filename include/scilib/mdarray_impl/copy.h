// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_COPY_H
#define SCILIB_MDARRAY_COPY_H

#include <experimental/mdspan>
#include <type_traits>

namespace Scilib {

namespace stdex = std::experimental;

// clang-format off
template <class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class T_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void
copy(stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    using size_type = stdex::extents<>::size_type;
    for (size_type i = 0; i < y.extent(0); ++i) {
        y(i) = x(i);
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type nrows_x,
          stdex::extents<>::size_type ncols_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type nrows_y,
          stdex::extents<>::size_type ncols_y,
          class T_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void copy(stdex::mdspan<T_x, 
                               stdex::extents<nrows_x, ncols_x>, 
                               Layout_x, 
                               Accessor_x> x,
                 stdex::mdspan<T_y, 
                               stdex::extents<nrows_y, ncols_y>, 
                               Layout_y, 
                               Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        for (size_type j = 0; j < y.extent(1); ++j) {
            y(i, j) = x(i, j);
        }
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void copy(stdex::mdspan<T_x, 
                               stdex::extents<n1_x, n2_x, n3_x>, 
                               Layout_x, 
                               Accessor_x> x,
                 stdex::mdspan<T_y, 
                               stdex::extents<n1_y, n2_y, n3_y>, 
                               Layout_y, 
                               Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        for (size_type j = 0; j < y.extent(1); ++j) {
            for (size_type k = 0; k < y.extent(2); ++k) {
                y(i, j, k) = x(i, j, k);
            }
        }
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          stdex::extents<>::size_type n4_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          stdex::extents<>::size_type n4_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void copy(stdex::mdspan<T_x, 
                               stdex::extents<n1_x, n2_x, n3_x, n4_x>, 
                               Layout_x, 
                               Accessor_x> x,
                 stdex::mdspan<T_y, 
                               stdex::extents<n1_y, n2_y, n3_y, n4_y>, 
                               Layout_y, 
                               Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));
    static_assert(x.static_extent(3) == y.static_extent(3));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        for (size_type j = 0; j < y.extent(1); ++j) {
            for (size_type k = 0; k < y.extent(2); ++k) {
                for (size_type l = 0; l < y.extent(3); ++l) {
                    y(i, j, k, l) = x(i, j, k, l);
                }
            }
        }
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          stdex::extents<>::size_type n4_x,
          stdex::extents<>::size_type n5_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          stdex::extents<>::size_type n4_y,
          stdex::extents<>::size_type n5_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void copy(stdex::mdspan<T_x,
                               stdex::extents<n1_x, n2_x, n3_x, n4_x, n5_x>,
                               Layout_x,
                               Accessor_x> x,
                 stdex::mdspan<T_y,
                               stdex::extents<n1_y, n2_y, n3_y, n4_y, n5_x>,
                               Layout_y,
                               Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));
    static_assert(x.static_extent(3) == y.static_extent(3));
    static_assert(x.static_extent(4) == y.static_extent(4));

    using size_type = stdex::extents<>::size_type;

    for (size_type i1 = 0; i1 < y.extent(0); ++i1) {
        for (size_type i2 = 0; i2 < y.extent(1); ++i2) {
            for (size_type i3 = 0; i3 < y.extent(2); ++i3) {
                for (size_type i4 = 0; i4 < y.extent(3); ++i4) {
                    for (size_type i5 = 0; i5 < y.extent(4); ++i5) {
                        y(i1, i2, i3, i4, i5) = x(i1, i2, i3, i4, i5);
                    }
                }
            }
        }
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          stdex::extents<>::size_type n4_x,
          stdex::extents<>::size_type n5_x,
          stdex::extents<>::size_type n6_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          stdex::extents<>::size_type n4_y,
          stdex::extents<>::size_type n5_y,
          stdex::extents<>::size_type n6_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void
copy(stdex::mdspan<T_x,
                   stdex::extents<n1_x, n2_x, n3_x, n4_x, n5_x, n6_x>,
                   Layout_x,
                   Accessor_x> x,
     stdex::mdspan<T_y,
                   stdex::extents<n1_y, n2_y, n3_y, n4_y, n5_x, n6_x>,
                   Layout_y,
                   Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));
    static_assert(x.static_extent(3) == y.static_extent(3));
    static_assert(x.static_extent(4) == y.static_extent(4));
    static_assert(x.static_extent(5) == y.static_extent(5));

    using size_type = stdex::extents<>::size_type;

    for (size_type i1 = 0; i1 < y.extent(0); ++i1) {
        for (size_type i2 = 0; i2 < y.extent(1); ++i2) {
            for (size_type i3 = 0; i3 < y.extent(2); ++i3) {
                for (size_type i4 = 0; i4 < y.extent(3); ++i4) {
                    for (size_type i5 = 0; i5 < y.extent(4); ++i5) {
                        for (size_type i6 = 0; i6 < y.extent(5); ++i6) {
                            y(i1, i2, i3, i4, i5, i6) =
                                x(i1, i2, i3, i4, i5, i6);
                        }
                    }
                }
            }
        }
    }
}

// clang-format off
template <class T_x,
          stdex::extents<>::size_type n1_x,
          stdex::extents<>::size_type n2_x,
          stdex::extents<>::size_type n3_x,
          stdex::extents<>::size_type n4_x,
          stdex::extents<>::size_type n5_x,
          stdex::extents<>::size_type n6_x,
          stdex::extents<>::size_type n7_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type n1_y,
          stdex::extents<>::size_type n2_y,
          stdex::extents<>::size_type n3_y,
          stdex::extents<>::size_type n4_y,
          stdex::extents<>::size_type n5_y,
          stdex::extents<>::size_type n6_y,
          stdex::extents<>::size_type n7_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void
copy(stdex::mdspan<T_x,
                   stdex::extents<n1_x, n2_x, n3_x, n4_x, n5_x, n6_x, n7_x>,
                   Layout_x,
                   Accessor_x> x,
     stdex::mdspan<T_y,
                   stdex::extents<n1_y, n2_y, n3_y, n4_y, n5_x, n6_x, n7_x>,
                   Layout_y,
                   Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    static_assert(x.static_extent(2) == y.static_extent(2));
    static_assert(x.static_extent(3) == y.static_extent(3));
    static_assert(x.static_extent(4) == y.static_extent(4));
    static_assert(x.static_extent(5) == y.static_extent(5));
    static_assert(x.static_extent(6) == y.static_extent(6));

    using size_type = stdex::extents<>::size_type;

    for (size_type i1 = 0; i1 < y.extent(0); ++i1) {
        for (size_type i2 = 0; i2 < y.extent(1); ++i2) {
            for (size_type i3 = 0; i3 < y.extent(2); ++i3) {
                for (size_type i4 = 0; i4 < y.extent(3); ++i4) {
                    for (size_type i5 = 0; i5 < y.extent(4); ++i5) {
                        for (size_type i6 = 0; i6 < y.extent(5); ++i6) {
                            for (size_type i7 = 0; i7 < y.extent(6); ++i7) {
                                y(i1, i2, i3, i4, i5, i6, i7) =
                                    x(i1, i2, i3, i4, i5, i6, i7);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_COPY_H
