// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_STATS_BITS_H
#define SCILIB_STATS_BITS_H

#include <experimental/mdspan>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <type_traits>
#include <cmath>

namespace Sci {
namespace Stats {
namespace stdex = std::experimental;

// Arithmetic mean.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto mean(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    using value_type = std::remove_cv_t<T>;
    value_type result =
        Sci::Linalg::sum(x) / static_cast<value_type>(x.extent(0));
    return result;
}

template <class T, class Layout, class Allocator>
inline T mean(const Sci::Vector<T, Layout, Allocator>& x)
{
    return mean(x.view());
}

// Median.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto median(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T>;

    Sci::Vector<value_type> xcopy(x);
    Sci::sort(xcopy.view());

    size_type n = (xcopy.extent(0) + 1) / 2;
    value_type med = xcopy(n);
    if (xcopy.extent(0) % 2 == 0) { // even
        n = (xcopy.extent(0) + 1) / 2 - 1;
        med = (med + xcopy(n)) / 2.0;
    }
    return med;
}

template <class T, class Layout, class Allocator>
inline T median(const Sci::Vector<T, Layout, Allocator>& x)
{
    return median(x.view());
}

// Variance.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto var(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T>;

    // Two-pass algorithm:
    value_type n = static_cast<value_type>(x.extent(0));
    value_type xmean = mean(x);
    value_type sum2 = value_type{0};

    for (size_type i = 0; i < x.extent(0); ++i) {
        sum2 += std::pow(x(i) - xmean, 2);
    }
    return sum2 / (n - 1.0);
}

template <class T, class Layout, class Allocator>
inline T var(const Sci::Vector<T, Layout, Allocator>& x)
{
    return var(x.view());
}

// Standard deviation.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto stddev(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    return std::sqrt(var(x));
}

template <class T, class Layout, class Allocator>
inline T stddev(const Sci::Vector<T, Layout, Allocator>& x)
{
    return stddev(x.view());
}

// Root-mean-square deviation.
template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto rms(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> x)
{
    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T>;

    value_type sum2 = value_type{0};
    for (size_type i = 0; i < x.extent(0); ++i) {
        sum2 += x(i) * x(i);
    }
    return std::sqrt(sum2 / x.extent(0));
}

template <class T, class Layout, class Allocator>
inline T rms(const Sci::Vector<T, Layout, Allocator>& x)
{
    return rms(x.view());
}

// Covariance.
template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline auto cov(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
                stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T>;

    value_type xmean = mean(x);
    value_type ymean = mean(y);
    value_type res = value_type{0};

    for (size_type i = 0; i < x.extent(0); ++i) {
        value_type a = x(i) - xmean;
        value_type b = y(i) - ymean;
        res += a * b / (x.extent(0) - T{1});
    }
    return res;
}

template <class T, class Layout, class Allocator>
inline T cov(const Sci::Vector<T, Layout, Allocator>& x,
             const Sci::Vector<T, Layout, Allocator>& y)
{
    return cov(x.view(), y.view());
}

} // namespace Stats
} // namespace Sci

#endif // SCILIB_STATS_BITS_H
