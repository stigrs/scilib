// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_AXPY_H
#define SCILIB_LINALG_BLAS1_AXPY_H

#include <gsl/gsl>

namespace Sci {
namespace Linalg {


template <class T_scalar,
          class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y> && std::is_integral_v<IndexType_x> &&
             std::is_integral_v<IndexType_y>)
inline void axpy(const T_scalar& scalar,
                 Kokkos::mdspan<T_x, Kokkos::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
                 Kokkos::mdspan<T_y, Kokkos::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));

    using index_type = IndexType_y;

    for (index_type i = 0; i < y.extent(0); ++i) {
        y[i] = scalar * x[i] + y[i];
    }
}

template <class T_scalar,
          class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Container_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y>
    requires(!std::is_const_v<T_y> && std::is_integral_v<IndexType_x> &&
             std::is_integral_v<IndexType_y>)
inline void
axpy(const T_scalar& scalar,
     const Sci::MDArray<T_x, Kokkos::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
     Sci::MDArray<T_y, Kokkos::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y)
{
    axpy(scalar, x.to_mdspan(), y.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_AXPY_H
