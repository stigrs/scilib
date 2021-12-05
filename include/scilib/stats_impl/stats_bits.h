// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_STATS_BITS_H
#define SCILIB_STATS_BITS_H

#include <experimental/mdspan>
#include <scilib/linalg.h>
#include <type_traits>

namespace Scilib {
namespace Stats {
namespace stdex = std::experimental;

// Arithmetic mean.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T mean(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    return Scilib::Linalg::sum(x) / static_cast<T>(x.extent(0));
}
} // namespace Stats
} // namespace Scilib

#endif // SCILIB_STATS_BITS_H
