// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
#define SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H

#include <cmath>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T, std::size_t ext_x, class Layout_x, class Accessor_x>
inline auto abs_sum(stdex::mdspan<T, stdex::extents<index, ext_x>, Layout_x, Accessor_x> x)
{
    using index_type = index;
    using value_type = std::remove_cv_t<T>;

    value_type result = 0;
    for (index_type i = 0; i < x.extent(0); ++i) {
        result += std::abs(x(i));
    }
    return result;
}

template <class T, class Layout, class Allocator>
inline T abs_sum(const Sci::Vector<T, Layout, Allocator>& x)
{
    return abs_sum(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
