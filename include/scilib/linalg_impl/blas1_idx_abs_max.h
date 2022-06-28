// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H
#define SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H

#include <experimental/mdspan>
#include <cmath>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T, std::size_t ext_x, class Layout_x, class Accessor_x>
inline std::size_t idx_abs_max(
    stdex::mdspan<T, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x>
        x)
{
    using size_type = std::size_t;
    using magn_type = std::remove_cv_t<decltype(std::abs(x(0)))>;

    size_type max_idx = 0;
    magn_type max_val = std::abs(x(0));
    for (size_type i = 0; i < x.extent(0); ++i) {
        if (max_val < std::abs(x(i))) {
            max_val = std::abs(x(i));
            max_idx = i;
        }
    }
    return max_idx;
}

template <class T, class Layout, class Allocator>
inline std::size_t idx_abs_max(const Sci::Vector<T, Layout, Allocator>& x)
{
    return idx_abs_max(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H
