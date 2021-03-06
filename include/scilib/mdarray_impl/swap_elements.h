// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SWAP_ELEMENTS_H
#define SCILIB_MDARRAY_SWAP_ELEMENTS_H

#include <gsl/gsl>
#include <type_traits>
#include <utility>


namespace Sci {

namespace stdex = std::experimental;

template <class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void
swap_elements(stdex::mdspan<T_x, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
              stdex::mdspan<T_y, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i = 0; i < x.extent(0); ++i) {
        std::swap(x(i), y(i));
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t nrows_x,
          std::size_t ncols_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t nrows_y,
          std::size_t ncols_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void swap_elements(
    stdex::mdspan<T_x, stdex::extents<IndexType_x, nrows_x, ncols_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<IndexType_y, nrows_y, ncols_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i = 0; i < x.extent(0); ++i) {
        for (index_type j = 0; j < x.extent(1); ++j) {
            std::swap(x(i, j), y(i, j));
        }
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t n1_x,
          std::size_t n2_x,
          std::size_t n3_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t n1_y,
          std::size_t n2_y,
          std::size_t n3_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void swap_elements(
    stdex::mdspan<T_x, stdex::extents<IndexType_x, n1_x, n2_x, n3_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<IndexType_y, n1_y, n2_y, n3_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));
    Expects(x.extent(2) == y.extent(2));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i = 0; i < x.extent(0); ++i) {
        for (index_type j = 0; j < x.extent(1); ++j) {
            for (index_type k = 0; k < x.extent(2); ++k) {
                std::swap(x(i, j, k), y(i, j, k));
            }
        }
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t n1_x,
          std::size_t n2_x,
          std::size_t n3_x,
          std::size_t n4_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t n1_y,
          std::size_t n2_y,
          std::size_t n3_y,
          std::size_t n4_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void swap_elements(
    stdex::mdspan<T_x, stdex::extents<IndexType_x, n1_x, n2_x, n3_x, n4_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<IndexType_y, n1_y, n2_y, n3_y, n4_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));
    Expects(x.extent(2) == y.extent(2));
    Expects(x.extent(3) == y.extent(3));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i = 0; i < x.extent(0); ++i) {
        for (index_type j = 0; j < x.extent(1); ++j) {
            for (index_type k = 0; k < x.extent(2); ++k) {
                for (index_type l = 0; l < x.extent(3); ++l) {
                    std::swap(x(i, j, k, l), y(i, j, k, l));
                }
            }
        }
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t n1_x,
          std::size_t n2_x,
          std::size_t n3_x,
          std::size_t n4_x,
          std::size_t n5_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t n1_y,
          std::size_t n2_y,
          std::size_t n3_y,
          std::size_t n4_y,
          std::size_t n5_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void swap_elements(
    stdex::
        mdspan<T_x, stdex::extents<IndexType_x, n1_x, n2_x, n3_x, n4_x, n5_x>, Layout_x, Accessor_x>
            x,
    stdex::
        mdspan<T_y, stdex::extents<IndexType_y, n1_y, n2_y, n3_y, n4_y, n5_y>, Layout_y, Accessor_y>
            y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));
    Expects(x.extent(2) == y.extent(2));
    Expects(x.extent(3) == y.extent(3));
    Expects(x.extent(4) == y.extent(4));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i1 = 0; i1 < x.extent(0); ++i1) {
        for (index_type i2 = 0; i2 < x.extent(1); ++i2) {
            for (index_type i3 = 0; i3 < x.extent(2); ++i3) {
                for (index_type i4 = 0; i4 < x.extent(3); ++i4) {
                    for (index_type i5 = 0; i5 < x.extent(4); ++i5) {
                        std::swap(x(i1, i2, i3, i4, i5), y(i1, i2, i3, i4, i5));
                    }
                }
            }
        }
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t n1_x,
          std::size_t n2_x,
          std::size_t n3_x,
          std::size_t n4_x,
          std::size_t n5_x,
          std::size_t n6_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t n1_y,
          std::size_t n2_y,
          std::size_t n3_y,
          std::size_t n4_y,
          std::size_t n5_y,
          std::size_t n6_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void
swap_elements(stdex::mdspan<T_x,
                            stdex::extents<IndexType_x, n1_x, n2_x, n3_x, n4_x, n5_x, n6_x>,
                            Layout_x,
                            Accessor_x> x,
              stdex::mdspan<T_y,
                            stdex::extents<IndexType_y, n1_y, n2_y, n3_y, n4_y, n5_y, n6_y>,
                            Layout_y,
                            Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));
    Expects(x.extent(2) == y.extent(2));
    Expects(x.extent(3) == y.extent(3));
    Expects(x.extent(4) == y.extent(4));
    Expects(x.extent(5) == y.extent(5));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i1 = 0; i1 < x.extent(0); ++i1) {
        for (index_type i2 = 0; i2 < x.extent(1); ++i2) {
            for (index_type i3 = 0; i3 < x.extent(2); ++i3) {
                for (index_type i4 = 0; i4 < x.extent(3); ++i4) {
                    for (index_type i5 = 0; i5 < x.extent(4); ++i5) {
                        for (index_type i6 = 0; i6 < x.extent(5); ++i6) {
                            std::swap(x(i1, i2, i3, i4, i5, i6), y(i1, i2, i3, i4, i5, i6));
                        }
                    }
                }
            }
        }
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t n1_x,
          std::size_t n2_x,
          std::size_t n3_x,
          std::size_t n4_x,
          std::size_t n5_x,
          std::size_t n6_x,
          std::size_t n7_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t n1_y,
          std::size_t n2_y,
          std::size_t n3_y,
          std::size_t n4_y,
          std::size_t n5_y,
          std::size_t n6_y,
          std::size_t n7_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline void
swap_elements(stdex::mdspan<T_x,
                            stdex::extents<IndexType_x, n1_x, n2_x, n3_x, n4_x, n5_x, n6_x, n7_x>,
                            Layout_x,
                            Accessor_x> x,
              stdex::mdspan<T_y,
                            stdex::extents<IndexType_x, n1_y, n2_y, n3_y, n4_y, n5_y, n6_y, n7_y>,
                            Layout_y,
                            Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));
    Expects(x.extent(1) == y.extent(1));
    Expects(x.extent(2) == y.extent(2));
    Expects(x.extent(3) == y.extent(3));
    Expects(x.extent(4) == y.extent(4));
    Expects(x.extent(5) == y.extent(5));
    Expects(x.extent(6) == y.extent(6));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i1 = 0; i1 < x.extent(0); ++i1) {
        for (index_type i2 = 0; i2 < x.extent(1); ++i2) {
            for (index_type i3 = 0; i3 < x.extent(2); ++i3) {
                for (index_type i4 = 0; i4 < x.extent(3); ++i4) {
                    for (index_type i5 = 0; i5 < x.extent(4); ++i5) {
                        for (index_type i6 = 0; i6 < x.extent(5); ++i6) {
                            for (index_type i7 = 0; i7 < x.extent(6); ++i7) {
                                std::swap(x(i1, i2, i3, i4, i5, i6, i7),
                                          y(i1, i2, i3, i4, i5, i6, i7));
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SWAP_ELEMENTS_H
