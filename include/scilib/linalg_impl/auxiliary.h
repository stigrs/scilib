// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_AUXILIARY_H
#define SCILIB_LINALG_AUXILIARY_H

#include <scilib/mdarray.h>
#include <experimental/mdspan>
#include <random>

namespace Scilib {
namespace Linalg {

//------------------------------------------------------------------------------
// Fill mdspan:

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline void fill(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v,
                 const T& value)
{
    Scilib::apply(v, [&](T& vi) { vi = value; });
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline void
fill(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
     const T& value)
{
    Scilib::apply(m, [&](T& mi) { mi = value; });
}

//------------------------------------------------------------------------------
// Limit array values:

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline void clip(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> a,
                 const T& a_min,
                 const T& a_max)
{
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < a.extent(0); ++i) {
        if (a(i) < a_min) {
            a(i) = a_min;
        }
        if (a_max < a(i)) {
            a(i) = a_max;
        }
    }
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline void
clip(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> a,
     const T& a_min,
     const T& a_max)
{
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < a.extent(0); ++i) {
        for (size_type j = 0; i < a.extent(1); ++i) {
            if (a(i, j) < a_min) {
                a(i, j) = a_min;
            }
            if (a_max < a(i)) {
                a(i, j) = a_max;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Find argmax, argmin, max, min, sum, and product of elements:

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline stdex::extents<>::size_type
argmax(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    size_type max_idx = 0;
    T max_val = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < max_val) {
            max_val = v(i);
            max_idx = i;
        }
    }
    return max_idx;
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline stdex::extents<>::size_type
argmin(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    size_type min_idx = 0;
    T min_val = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (min_val < v(i)) {
            min_val = v(i);
            min_idx = i;
        }
    }
    return min_idx;
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline T max(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    T result = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (result < v(i)) {
            result = v(i);
        }
    }
    return result;
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline T min(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    T result = v(0);
    for (size_type i = 0; i < v.extent(0); ++i) {
        if (v(i) < result) {
            result = v(i);
        }
    }
    return result;
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline T sum(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    T result = 0;
    for (size_type i = 0; i < v.extent(0); ++i) {
        result += v(i);
    }
    return result;
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline T prod(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    T result = 1;
    for (size_type i = 0; i < v.extent(0); ++i) {
        result *= v(i);
    }
    return result;
}

//------------------------------------------------------------------------------
// Create special vectors and matrices:

// clang-format off
template <class M, class... Args>
    requires MDArray_type<M>
inline M zeros(Args... args)
// clang-format on
{
    static_assert(M::N_dim == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{0};
    return res;
}

// clang-format off
template <class M, class... Args>
    requires MDArray_type<M>
inline M ones(Args... args)
// clang-format on
{
    static_assert(M::N_dim == sizeof...(Args));
    using value_type = typename M::value_type;

    M res(args...);
    res = value_type{1};
    return res;
}

template <class T = double>
inline Matrix<T> identity(std::size_t n)
{
    Matrix<T> res(n, n);
    res = T{0};
    auto res_diag = Scilib::diag(res.view());
    for (std::size_t i = 0; i < res_diag.extent(0); ++i) {
        res_diag(i) = T{1};
    }
    return res;
}

// Create a random vector from a normal distribution with zero mean and unit
// variance.
template <class T = double>
inline Scilib::Vector<T> randn(std::size_t n)
{
    Scilib::Vector<T> res(n);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<T> nd{};

    for (auto& x : res) {
        x = nd(gen);
    }
    return res;
}

// Create a random matrix from a normal distribution with zero mean and unit
// variance.
template <class T = double>
inline Scilib::Matrix<T> randn(std::size_t nr, std::size_t nc)
{
    Scilib::Matrix<T> res(nr, nc);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<T> nd{};

    for (auto& x : res) {
        x = nd(gen);
    }
    return res;
}

// Create a random vector from a uniform real distribution on the
// interval [0, 1).
inline Scilib::Vector<double> randu(std::size_t n)
{
    Scilib::Vector<double> res(n);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_real_distribution<> ur{};

    for (auto& x : res) {
        x = ur(gen);
    }
    return res;
}

// Create a random matrix from a uniform real distribution on the
// interval [0, 1).
inline Scilib::Matrix<double> randu(std::size_t nr, std::size_t nc)
{
    Scilib::Matrix<double> res(nr, nc);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_real_distribution<> ur{};

    for (auto& x : res) {
        x = ur(gen);
    }
    return res;
}

// Create a random vector from a uniform integer distribution on the
// interval [0, 1].
inline Scilib::Vector<int> randi(std::size_t n)
{
    Scilib::Vector<int> res(n);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_int_distribution<> ui{};

    for (auto& x : res) {
        x = ui(gen);
    }
    return res;
}

// Create a random matrix from a uniform integer distribution on the
// interval [0, 1].
inline Scilib::Matrix<int> randi(std::size_t nr, std::size_t nc)
{
    Scilib::Matrix<int> res(nr, nc);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_int_distribution<> ui{};

    for (auto& x : res) {
        x = ui(gen);
    }
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_AUXILIARY_H
