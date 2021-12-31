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

namespace Sci {
namespace Integrate {

namespace stdex = std::experimental;

// Integrate function values over a non-uniform grid using the
// trapezoidal rule.
template <class T_scalar,
          class T_x,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto trapz(T_scalar xlo,
                  T_scalar xup,
                  stdex::mdspan<T_x, stdex::extents<ext>, Layout, Accessor> x)
{
    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T_x>;

    const value_type step = std::abs(xup - xlo) / (x.extent(0) - 1);
    value_type ans = value_type{0};

    for (size_type i = 1; i < x.extent(0); ++i) {
        ans += 0.5 * (x(i) + x(i - 1));
    }
    return ans *= step;
}

template <class T, class Layout>
inline auto trapz(T xlo, T xup, const Sci::Vector<T, Layout>& x)
{
    return trapz(xlo, xup, x.view());
}

} // namespace Integrate
} // namespace Sci

#endif // SCILIB_INTEGRATE_IMPL_TRAPZ_H
