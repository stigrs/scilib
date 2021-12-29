// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_TRACE_H
#define SCILIB_LINALG_TRACE_H

#include <experimental/mdspan>
#include <type_traits>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout_m,
          class Accessor_m>
    requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
inline T
trace(stdex::mdspan<T, stdex::extents<ext, ext>, Layout_m, Accessor_m> m)
// clang-format on
{
    return Scilib::Linalg::sum(Scilib::diag(m));
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_TRACE_H
