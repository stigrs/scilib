// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_IMPL_TRAPZ_H
#define SCILIB_INTEGRATE_IMPL_TRAPZ_H

#include "../mdarray.h"
#include <cmath>
#include <type_traits>

namespace Sci {
namespace Integrate {

namespace Mdspan = std::experimental;

// Integrate function values over a non-uniform grid using the
// trapezoidal rule.
template <class T_scalar, class T_x, std::size_t ext, class Layout, class Accessor>
inline auto trapz(T_scalar xlo,
                  T_scalar xup,
                  Mdspan::mdspan<T_x, Mdspan::extents<index, ext>, Layout, Accessor> x)
{
    using index_type = index;
    using value_type = std::remove_cv_t<T_x>;

    const value_type step = std::abs(xup - xlo) / (x.extent(0) - 1);
    value_type ans = value_type{0};

    for (index_type i = 1; i < x.extent(0); ++i) {
        ans += 0.5 * (x[i] + x[i - 1]);
    }
    return ans *= step;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
inline auto
trapz(T xlo, T xup, const Sci::MDArray<T, Mdspan::extents<IndexType, ext>, Layout, Container>& x)
{
    return trapz(xlo, xup, x.to_mdspan());
}

} // namespace Integrate
} // namespace Sci

#endif // SCILIB_INTEGRATE_IMPL_TRAPZ_H
