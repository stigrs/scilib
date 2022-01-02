// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H
#define SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H

#include <experimental/mdspan>
#include <cmath>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x>
inline stdex::extents<>::size_type
idx_abs_min(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x)
{
    using size_type = stdex::extents<>::size_type;
    using magn_type = std::remove_cv_t<decltype(std::abs(x(0)))>;

    size_type min_idx = 0;
    magn_type min_val = std::abs(x(0));
    for (size_type i = 0; i < x.extent(0); ++i) {
        if (std::abs(x(i)) < min_val) {
            min_val = std::abs(x(i));
            min_idx = i;
        }
    }
    return min_idx;
}

template <class T, class Layout, class Allocator>
inline stdex::extents<>::size_type
idx_abs_min(const Sci::Vector<T, Layout, Allocator>& x)
{
    return idx_abs_min(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H
