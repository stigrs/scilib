// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H
#define SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H

#include <cmath>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace Mdspan = std::experimental;

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline IndexType idx_abs_min(Mdspan::mdspan<T, Mdspan::extents<IndexType, ext>, Layout, Accessor> x)
{
    using index_type = IndexType;
    using magn_type = std::remove_cv_t<decltype(std::abs(x[0]))>;

    index_type min_idx = 0;
    magn_type min_val = std::abs(x[0]);
    for (index_type i = 0; i < x.extent(0); ++i) {
        if (std::abs(x[i]) < min_val) {
            min_val = std::abs(x[i]);
            min_idx = i;
        }
    }
    return min_idx;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline IndexType
idx_abs_min(const Sci::MDArray<T, Mdspan::extents<IndexType, ext>, Layout, Container>& x)
{
    return idx_abs_min(x.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_IDX_ABS_MIN_H
