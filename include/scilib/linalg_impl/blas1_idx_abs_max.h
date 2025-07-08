// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H
#define SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {


template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline IndexType
idx_abs_max(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return Kokkos::Experimental::linalg::vector_idx_abs_max(x.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_IDX_ABS_MAX_H
