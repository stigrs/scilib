// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_MATRIX_DIAG_H
#define SCILIB_MDARRAY_MATRIX_DIAG_H

#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/type_aliases.h>
#include <experimental/mdspan>
#include <vector>
#include <array>

namespace Scilib {

// Non-owing view of matrix diagonal.
template <class T>
class diag {
public:
    using value_type = T;
    using size_type = std::experimental::extents<>::size_type;

    diag() = delete;

    diag(Matrix_view<T> m)
        : span{m.data(),
               {stdex::dextents<1>{m.extent(0)},
                std::array<std::size_t, 1>{m.stride(0) + 1}}}
    {
        static_assert(m.static_extent(0) == m.static_extent(1));
    }

    diag& operator=(const Vector<T>& v)
    {
        static_assert(v.static_extent(0) == span.static_extent(0));
        for (size_type i = 0; i < v.size(); ++i) {
            span(i) = v(i);
        }
        return *this;
    }

    template <stdex::extents<>::size_type ext, class Layout, class Accessor>
    diag& operator=(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
    {
        static_assert(v.static_extent(0) == span.static_extent(0));
        for (size_type i = 0; i < v.extent(0); ++i) {
            span(i) = v(i);
        }
        return *this;
    }

    diag& operator=(const std::vector<T>& v)
    {
        static_assert(v.static_extent(0) == span.static_extent(0));
        for (size_type i = 0; i < v.size(); ++i) {
            span(i) = v(i);
        }
        return *this;
    }

    ~diag() = default;

    constexpr auto& operator()(size_type i) noexcept { return span(i); }
    constexpr const auto& operator()(size_type i) const noexcept
    {
        return span(i);
    }

    constexpr size_type size() const noexcept { return span.size(); }

    constexpr size_type extent(size_type dim = 0) const noexcept
    {
        return span.extent(dim);
    }

    constexpr T* data() noexcept { return span.data(); }
    constexpr const T* data() const noexcept { return span.data(); }

    constexpr auto view() noexcept { return span; }
    constexpr const auto view() const noexcept { return span; }

    // Apply f(x) for every element x.
    template <class F>
    diag& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(span(i));
        }
        return *this;
    }

    // Apply f(x, val) for every element x.
    template <class F>
    diag& apply(F f, const T& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(span(i));
        }
        return *this;
    }

    diag& operator=(const T& value) noexcept
    {
        return apply([&](T& a) { a = value; });
    }

    diag& operator+=(const T& value) noexcept
    {
        return apply([&](T& a) { a += value; });
    }

    diag& operator-=(const T& value) noexcept
    {
        return apply([&](T& a) { a -= value; });
    }

    diag& operator*=(const T& value) noexcept
    {
        return apply([&](T& a) { a *= value; });
    }

    diag& operator/=(const T& value) noexcept
    {
        return apply([&](T& a) { a /= value; });
    }

    diag& operator%=(const T& value) noexcept
    {
        return apply([&](T& a) { a %= value; });
    }

private:
    Subvector_view<T> span;
};

} // namespace Scilib

#endif // SCILIB_MDARRAY_MATRIX_DIAG_H
