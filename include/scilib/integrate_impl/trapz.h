// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_IMPL_TRAPZ_H
#define SCILIB_INTEGRATE_IMPL_TRAPZ_H

#include <experimental/mdspan>
#include <type_traits>
#include <cmath>

namespace Scilib {
namespace Integrate {

namespace stdex = std::experimental;

// Integrate function values over a non-uniform grid using the
// trapezoidal rule.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T
trapz(T xlo, T xup, stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    const T step = std::abs(xup - xlo) / (x.extent(0) - 1);
    T ans = T{0};

    for (size_type i = 1; i < x.extent(0); ++i) {
        ans += 0.5 * (x(i) + x(i - 1));
    }
    return ans *= step;
}

} // namespace Integrate
} // namespace Scilib

#endif // SCILIB_INTEGRATE_IMPL_TRAPZ_H
