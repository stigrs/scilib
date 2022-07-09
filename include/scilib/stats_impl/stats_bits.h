// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_STATS_BITS_H
#define SCILIB_STATS_BITS_H

#include "../linalg.h"
#include "../mdarray.h"
#include <cmath>
#include <type_traits>

namespace Sci {
namespace Stats {
namespace stdex = std::experimental;

// Arithmetic mean.
template <class T, std::size_t ext, class Layout, class Accessor>
inline auto mean(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> x)
{
    using value_type = std::remove_cv_t<T>;
    value_type result = Sci::Linalg::sum(x) / static_cast<value_type>(x.extent(0));
    return result;
}

template <class T, class Layout, class Container>
inline T mean(const Sci::Vector<T, Layout, Container>& x)
{
    return mean(x.view());
}

// Median.
template <class T, std::size_t ext, class Layout, class Accessor>
inline auto median(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> x)
{
    using index_type = index;
    using value_type = std::remove_cv_t<T>;

    Sci::Vector<value_type> xcopy(x);
    Sci::sort(xcopy.view());

    index_type n = (xcopy.extent(0) + 1) / 2;
    value_type med = xcopy(n);
    if (xcopy.extent(0) % 2 == 0) { // even
        n = (xcopy.extent(0) + 1) / 2 - 1;
        med = (med + xcopy(n)) / 2.0;
    }
    return med;
}

template <class T, class Layout, class Container>
inline T median(const Sci::Vector<T, Layout, Container>& x)
{
    return median(x.view());
}

// Variance.
template <class T, std::size_t ext, class Layout, class Accessor>
inline auto var(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> x)
{
    using index_type = index;
    using value_type = std::remove_cv_t<T>;

    // Two-pass algorithm:
    value_type n = static_cast<value_type>(x.extent(0));
    value_type xmean = mean(x);
    value_type sum2 = value_type{0};

    for (index_type i = 0; i < x.extent(0); ++i) {
        sum2 += std::pow(x(i) - xmean, 2);
    }
    return sum2 / (n - 1.0);
}

template <class T, class Layout, class Container>
inline T var(const Sci::Vector<T, Layout, Container>& x)
{
    return var(x.view());
}

// Standard deviation.
template <class T, std::size_t ext, class Layout, class Accessor>
inline auto stddev(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> x)
{
    return std::sqrt(var(x));
}

template <class T, class Layout, class Container>
inline T stddev(const Sci::Vector<T, Layout, Container>& x)
{
    return stddev(x.view());
}

// Root-mean-square deviation.
template <class T, std::size_t ext, class Layout, class Accessor>
inline auto rms(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> x)
{
    using index_type = index;
    using value_type = std::remove_cv_t<T>;

    value_type sum2 = value_type{0};
    for (index_type i = 0; i < x.extent(0); ++i) {
        sum2 += x(i) * x(i);
    }
    return std::sqrt(sum2 / x.extent(0));
}

template <class T, class Layout, class Container>
inline T rms(const Sci::Vector<T, Layout, Container>& x)
{
    return rms(x.view());
}

// Covariance.
template <class T,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
inline auto cov(stdex::mdspan<T, stdex::extents<index, ext_x>, Layout_x, Accessor_x> x,
                stdex::mdspan<T, stdex::extents<index, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using index_type = index;
    using value_type = std::remove_cv_t<T>;

    value_type xmean = mean(x);
    value_type ymean = mean(y);
    value_type res = value_type{0};

    for (index_type i = 0; i < x.extent(0); ++i) {
        value_type a = x(i) - xmean;
        value_type b = y(i) - ymean;
        res += a * b / (x.extent(0) - T{1});
    }
    return res;
}

template <class T, class Layout, class Container>
inline T cov(const Sci::Vector<T, Layout, Container>& x, const Sci::Vector<T, Layout, Container>& y)
{
    return cov(x.view(), y.view());
}

} // namespace Stats
} // namespace Sci

#endif // SCILIB_STATS_BITS_H
