// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_AUXILIARY_H
#define SCILIB_LINALG_AUXILIARY_H

#include <cassert>
#include <random>
#include <type_traits>

namespace Sci {
namespace Linalg {

//--------------------------------------------------------------------------------------------------
// Fill mdspan:

template <class T, std::size_t ext, class Layout, class Accessor>
inline void fill(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v,
                 const T& value)
{
    Sci::apply(v, [&](T& vi) { vi = value; });
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
inline void fill(stdex::mdspan<T, stdex::extents<std::size_t, nrows, ncols>, Layout, Accessor> m,
                 const T& value)
{
    Sci::apply(m, [&](T& mi) { mi = value; });
}

template <class T, class Extents, class Layout, class Allocator>
    requires Extents_has_rank<Extents>
inline void fill(Sci::MDArray<T, Extents, Layout, Allocator>& m, const T& value)
{
    static_assert(Extents::rank() <= 2);
    fill(m.view(), value);
}

//--------------------------------------------------------------------------------------------------
// Limit array values:

template <class T, std::size_t ext, class Layout, class Accessor>
inline void clip(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> a,
                 const T& a_min,
                 const T& a_max)
{
    using size_type = std::size_t;

    for (size_type i = 0; i < a.extent(0); ++i) {
        if (a(i) < a_min) {
            a(i) = a_min;
        }
        if (a_max < a(i)) {
            a(i) = a_max;
        }
    }
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
inline void clip(stdex::mdspan<T, stdex::extents<std::size_t, nrows, ncols>, Layout, Accessor> a,
                 const T& a_min,
                 const T& a_max)
{
    using size_type = std::size_t;

    for (size_type i = 0; i < a.extent(0); ++i) {
        for (size_type j = 0; i < a.extent(1); ++i) {
            if (a(i, j) < a_min) {
                a(i, j) = a_min;
            }
            if (a_max < a(i, j)) {
                a(i, j) = a_max;
            }
        }
    }
}

template <class T, class Extents, class Layout, class Allocator>
inline void clip(Sci::MDArray<T, Extents, Layout, Allocator>& a, const T& a_min, const T& a_max)
{
    clip(a.view(), a_min, a_max);
}

//--------------------------------------------------------------------------------------------------
// Find argmax, argmin, max, min, sum, and product of elements:

template <class T, std::size_t ext, class Layout, class Accessor>
inline std::size_t argmax(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    size_type max_idx = 0;
    value_type max_val = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) > max_val) {
            max_val = v(i);
            max_idx = i;
        }
    }
    return max_idx;
}

template <class T, class Layout, class Allocator>
inline std::size_t argmax(const Sci::Vector<T, Layout, Allocator>& v)
{
    return argmax(v.view());
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline std::size_t argmin(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    size_type min_idx = 0;
    value_type min_val = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < min_val) {
            min_val = v(i);
            min_idx = i;
        }
    }
    return min_idx;
}

template <class T, class Layout, class Allocator>
inline std::size_t argmin(const Sci::Vector<T, Layout, Allocator>& v)
{
    return argmin(v.view());
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto max(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    value_type result = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) > result) {
            result = v(i);
        }
    }
    return result;
}

template <class T, class Layout, class Allocator>
inline T max(const Sci::Vector<T, Layout, Allocator>& v)
{
    return max(v.view());
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto min(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    value_type result = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < result) {
            result = v(i);
        }
    }
    return result;
}

template <class T, class Layout, class Allocator>
inline T min(const Sci::Vector<T, Layout, Allocator>& v)
{
    return min(v.view());
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto sum(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    value_type result = 0;
    for (size_type i = 0; i < v.extent(0); ++i) {
        result += v(i);
    }
    return result;
}

template <class T, class Layout, class Allocator>
inline T sum(const Sci::Vector<T, Layout, Allocator>& v)
{
    return sum(v.view());
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto prod(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v)
{
    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    value_type result = 1;
    for (size_type i = 0; i < v.extent(0); ++i) {
        result *= v(i);
    }
    return result;
}

template <class T, class Layout, class Allocator>
inline T prod(const Sci::Vector<T, Layout, Allocator>& v)
{
    return prod(v.view());
}

//--------------------------------------------------------------------------------------------------
// Create special vectors and matrices:

template <class M, class... Args>
    requires MDArray_type<M>
inline M zeros(Args... args)
{
    static_assert(M::rank() == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{0};
    return res;
}

template <class M, class... Args>
    requires MDArray_type<M>
inline M ones(Args... args)
{
    static_assert(M::rank() == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{1};
    return res;
}

template <class M = Sci::Matrix<double>>
    requires MDArray_type<M>
inline M identity(std::size_t n)
{
    static_assert(M::rank() == 2);

    using value_type = typename M::value_type;
    using size_type = typename M::size_type;

    M res(n, n);
    auto res_diag = Sci::diag(res.view());
    for (size_type i = 0; i < res_diag.extent(0); ++i) {
        res_diag(i) = value_type{1};
    }
    return res;
}

// Create a random MDArray from a normal distribution with zero mean and unit
// variance.
template <class M, class... Args>
    requires(MDArray_type<M>&& std::is_floating_point_v<typename M::value_type>)
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
    requires(MDArray_type<M>&& std::is_floating_point_v<typename M::value_type>)
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
    requires(MDArray_type<M>&& std::is_integral_v<typename M::value_type>)
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

template <class T, class Layout, class Allocator>
    requires(std::is_floating_point_v<T>)
Sci::Vector<T, Layout, Allocator> linspace(T start, T stop, int num = 50)
{
    assert(stop > start);
    T step_size = (stop - start) / (num - 1);
    T value = start;

    Sci::Vector<T, Layout, Allocator> res(num);

    res(0) = start;
    for (int i = 1; i < num; ++i) {
        value += step_size;
        res(i) = value;
    }
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_AUXILIARY_H
