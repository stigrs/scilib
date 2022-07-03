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

namespace stdex = std::experimental;

template <class T, std::size_t ext, class Layout_m, class Accessor_m>
    requires(std::is_arithmetic_v<std::remove_cv_t<T>>)
inline auto trace(stdex::mdspan<T, stdex::extents<index, ext, ext>, Layout_m, Accessor_m> m)
{
    return Sci::Linalg::sum(Sci::diag(m));
}

template <class T, class Layout, class Allocator>
inline T trace(const Sci::Matrix<T, Layout, Allocator>& m)
{
    assert(m.extent(0) == m.extent(1));
    return trace(m.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_TRACE_H
