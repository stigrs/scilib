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
#include <gsl/gsl>
#include <type_traits>

namespace Sci {
namespace Stats {

// Arithmetic mean.
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto mean(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    using value_type = std::remove_cv_t<T>;
    value_type result = Sci::Linalg::sum(x) / static_cast<value_type>(x.extent(0));
    return result;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T mean(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return mean(x.to_mdspan());
}

// Median.
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto median(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    Sci::Vector<value_type> xcopy(x);
    Sci::sort(xcopy.to_mdspan());

    index_type n = (xcopy.extent(0) + 1) / 2;
    value_type med = xcopy[n];
    if (xcopy.extent(0) % 2 == 0) { // even
        n = (xcopy.extent(0) + 1) / 2 - 1;
        med = (med + xcopy[n]) / 2.0;
    }
    return med;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T median(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return median(x.to_mdspan());
}

// Variance.
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto var(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    // Two-pass algorithm:
    value_type n = static_cast<value_type>(x.extent(0));
    value_type xmean = mean(x);
    value_type sum2 = value_type{0};

    for (index_type i = 0; i < x.extent(0); ++i) {
        sum2 += std::pow(x[i] - xmean, 2);
    }
    return sum2 / (n - 1.0);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T var(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return var(x.to_mdspan());
}

// Standard deviation.
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto stddev(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    return std::sqrt(var(x));
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T stddev(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return stddev(x.to_mdspan());
}

// Root-mean-square deviation.
template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto rms(Kokkos::mdspan<T, Kokkos::extents<IndexType, ext>, Layout, Accessor> x)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    value_type sum2 = value_type{0};
    for (index_type i = 0; i < x.extent(0); ++i) {
        sum2 += x[i] * x[i];
    }
    return std::sqrt(sum2 / x.extent(0));
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T rms(const Sci::MDArray<T, Kokkos::extents<IndexType, ext>, Layout, Container>& x)
{
    return rms(x.to_mdspan());
}

// Covariance.
template <class T,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline auto cov(Kokkos::mdspan<T, Kokkos::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
                Kokkos::mdspan<T, Kokkos::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    Expects(x.extent(0) == y.extent(0));

    using index_type = std::common_type_t<IndexType_x, IndexType_y>;
    using value_type = std::remove_cv_t<T>;

    value_type xmean = mean(x);
    value_type ymean = mean(y);
    value_type res = value_type{0};

    for (index_type i = 0; i < x.extent(0); ++i) {
        value_type a = x[i] - xmean;
        value_type b = y[i] - ymean;
        res += a * b / (x.extent(0) - T{1});
    }
    return res;
}

template <class T,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Container_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline T cov(const Sci::MDArray<T, Kokkos::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
             const Sci::MDArray<T, Kokkos::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y)
{
    return cov(x.to_mdspan(), y.to_mdspan());
}

} // namespace Stats
} // namespace Sci

#endif // SCILIB_STATS_BITS_H
