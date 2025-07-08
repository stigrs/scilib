// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_SCALED_H
#define SCILIB_LINALG_SCALED_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class ScalingFactorType, class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
scaled(ScalingFactorType scaling_factor, const Sci::MDArray<T, Extents, Layout, Container>& a)
{
    return Sci::MDArray<T, Extents, Layout, Container>(
        Kokkos::Experimental::linalg::scaled(scaling_factor, a.to_mdspan()));
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_SCALED_H
