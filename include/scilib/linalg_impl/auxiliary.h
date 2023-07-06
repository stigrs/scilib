// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_AUXILIARY_H
#define SCILIB_LINALG_AUXILIARY_H

#include <gsl/gsl>
#include <random>
#include <type_traits>

namespace Sci {
namespace Linalg {

//--------------------------------------------------------------------------------------------------
// Fill mdspan:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void fill(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v,
                 const T& value)
{
    Sci::apply(v, [&](T& vi) { vi = value; });
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void fill(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> m,
                 const T& value)
{
    Sci::apply(m, [&](T& mi) { mi = value; });
}

template <class T, class Extents, class Layout, class Container>
inline void fill(Sci::MDArray<T, Extents, Layout, Container>& m, const T& value)
{
    static_assert(Extents::rank() <= 2);
    fill(m.view(), value);
}

//--------------------------------------------------------------------------------------------------
// Limit array values:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void clip(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> a,
                 const T& a_min,
                 const T& a_max)
{
    using index_type = IndexType;

    for (index_type i = 0; i < a.extent(0); ++i) {
        if (a(i) < a_min) {
            a(i) = a_min;
        }
        if (a_max < a(i)) {
            a(i) = a_max;
        }
    }
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void clip(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> a,
                 const T& a_min,
                 const T& a_max)
{
    using index_type = IndexType;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; i < a.extent(1); ++i) {
            if (a(i, j) < a_min) {
                a(i, j) = a_min;
            }
            if (a_max < a(i, j)) {
                a(i, j) = a_max;
            }
        }
    }
}

template <class T, class Extents, class Layout, class Container>
inline void clip(Sci::MDArray<T, Extents, Layout, Container>& a, const T& a_min, const T& a_max)
{
    clip(a.view(), a_min, a_max);
}

//--------------------------------------------------------------------------------------------------
// Find argmax, argmin, max, min, sum, and product of elements:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline std::size_t argmax(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    index_type max_idx = 0;
    value_type max_val = v(0);
    for (index_type i = 0; i < v.extent(0); ++i) {
        if (v(i) > max_val) {
            max_val = v(i);
            max_idx = i;
        }
    }
    return max_idx;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline std::size_t
argmax(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return argmax(v.view());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline std::size_t argmin(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    index_type min_idx = 0;
    value_type min_val = v(0);
    for (index_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < min_val) {
            min_val = v(i);
            min_idx = i;
        }
    }
    return min_idx;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline std::size_t
argmin(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return argmin(v.view());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto max(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    value_type result = v(0);
    for (index_type i = 0; i < v.extent(0); ++i) {
        if (v(i) > result) {
            result = v(i);
        }
    }
    return result;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T max(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return max(v.view());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto min(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    value_type result = v(0);
    for (index_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < result) {
            result = v(i);
        }
    }
    return result;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T min(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return min(v.view());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto sum(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    value_type result = 0;
    for (index_type i = 0; i < v.extent(0); ++i) {
        result += v(i);
    }
    return result;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T sum(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return sum(v.view());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto prod(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    value_type result = 1;
    for (index_type i = 0; i < v.extent(0); ++i) {
        result *= v(i);
    }
    return result;
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline T prod(const Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    return prod(v.view());
}

//--------------------------------------------------------------------------------------------------
// Create special vectors and matrices:

// clang-format off
template <class M, class... Args>
    requires(__Detail::Is_mdarray_v<M> && M::rank() == sizeof...(Args))
// clang-format on
inline M zeros(Args... args)
{
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{0};
    return res;
}

// clang-format off
template <class M, class... Args>
    requires(__Detail::Is_mdarray_v<M> && M::rank() == sizeof...(Args))
// clang-format on
inline M ones(Args... args)
{
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{1};
    return res;
}

// clang-format off
template <class M = Sci::Matrix<double>>
    requires(__Detail::Is_mdarray_v<M> && M::rank() == 2)
// clang-format on
inline M identity(std::size_t n)
{
    using value_type = typename M::value_type;
    using index_type = typename M::index_type;

    M res(n, n);
    auto res_diag = Sci::diag(res.view());
    for (index_type i = 0; i < res_diag.extent(0); ++i) {
        res_diag(i) = value_type{1};
    }
    return res;
}

// Create a random MDArray from a normal distribution with zero mean and unit
// variance.
template <class M, class... Args>
    requires(__Detail::Is_mdarray_v<M>&& std::is_floating_point_v<typename M::value_type>)
inline M randn(Args... args)
{
    static_assert(M::rank() == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<value_type> nd{};

    for (auto& x : res) {
        x = nd(gen);
    }
    return res;
}

// Create a random MDArray from a uniform real distribution on the
// interval [0, 1).
template <class M, class... Args>
    requires(__Detail::Is_mdarray_v<M>&& std::is_floating_point_v<typename M::value_type>)
inline M randu(Args... args)
{
    static_assert(M::rank() == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_real_distribution<value_type> ur{};

    for (auto& x : res) {
        x = ur(gen);
    }
    return res;
}

// Create a random MDArray from a uniform integer distribution on the
// interval [0, 1].
template <class M, class... Args>
    requires(__Detail::Is_mdarray_v<M>&& std::is_integral_v<typename M::value_type>)
inline M randi(Args... args)
{
    static_assert(M::rank() == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_int_distribution<value_type> ui{};

    for (auto& x : res) {
        x = ui(gen);
    }
    return res;
}

template <class T, class Layout>
    requires(std::is_floating_point_v<T>)
Sci::Vector<T, Layout> linspace(T start, T stop, int num = 50)
{
    Expects(stop > start);
    T step_size = (stop - start) / (num - 1);
    T value = start;

    Sci::Vector<T, Layout> res(num);

    res(0) = start;
    for (int i = 1; i < num; ++i) {
        value += step_size;
        res(i) = value;
    }
    return res;
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
void to_lower_triangular(
    stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> a)
{
    Expects(a.extent(0) == a.extent(1));

    using index_type = IndexType;
    using value_type = std::remove_cv_t<T>;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = i + 1; j < a.extent(0); ++j) {
            a(i, j) = value_type{0};
        }
    }
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
void to_lower_triangular(
    Sci::MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& a)
{
    to_lower_triangular(a.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_AUXILIARY_H
