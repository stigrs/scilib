// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_TRACE_H
#define SCILIB_LINALG_TRACE_H

#include <cassert>
#include <type_traits>

namespace Sci {
namespace Linalg {


template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto trace(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext, ext>, Layout, Accessor> m)
{
    return Sci::Linalg::sum(Sci::diag(m));
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T trace(const Sci::MDArray<T, Kokkos::extents<IndexType, ext, ext>, Layout, Container>& m)
{
    Expects(m.extent(0) == m.extent(1));
    return trace(m.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_TRACE_H
