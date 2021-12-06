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
#include <cmath>

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

// Median.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T median(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    Scilib::Vector<T> xcopy(x);
    Scilib::Linalg::sort(xcopy.view());

    size_type n = (xcopy.extent(0) + 1) / 2;
    T med = xcopy(n);
    if (xcopy.extent(0) % 2 == 0) { // even
        n = (xcopy.extent(0) + 1) / 2 - 1;
        med = (med + xcopy(n)) / 2.0;
    }
    return med;
}

// Variance.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T var(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    // Two-pass algorithm:
    T n = static_cast<T>(x.extent(0));
    T xmean = mean(x);
    T sum2 = T{0};

    for (size_type i = 0; i < x.extent(0); ++i) {
        sum2 += std::pow(x(i) - xmean, 2);
    }
    return sum2 / (n - 1.0);
}

// Standard deviation.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T stddev(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    return std::sqrt(var(x));
}

// Root-mean-square deviation.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
    requires std::is_floating_point_v<T> 
inline T rms(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    T sum2 = T{0};
    for (size_type i = 0; i < x.extent(0); ++i) {
        sum2 += x(i) * x(i);
    }
    return std::sqrt(sum2 / x.extent(0));
}

// Covariance.
// clang-format off
template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
    requires std::is_floating_point_v<T> 
inline T cov(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x, 
             stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    static_assert(x.static_extent(0) == y.static_extent(0));

    T xmean = mean(x);
    T ymean = mean(y);
    T res = T{0};

    for (size_type i = 0; i < x.extent(0); ++i) {
        T a = x(i) - xmean;
        T b = y(i) - ymean;
        res += a * b / (x.extent(0) - T{1});
    }
    return res;
}

} // namespace Stats
} // namespace Scilib

#endif // SCILIB_STATS_BITS_H
