// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_ADD_H
#define SCILIB_LINALG_BLAS1_ADD_H

#include <experimental/mdspan>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y,
          stdex::extents<>::size_type ext_z,
          class Layout_z,
          class Accessor_z>
inline void add(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
                stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y,
                stdex::mdspan<T, stdex::extents<ext_z>, Layout_z, Accessor_z> z)
{
    static_assert(x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < z.extent(0); ++i) {
        z(i) = x(i) + y(i);
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS1_ADD_H
