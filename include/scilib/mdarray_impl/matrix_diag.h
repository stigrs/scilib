// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray_impl/type_aliases.h>
#include <scilib/mdarray_impl/matrix.h>
#include <scilib/mdarray_impl/vector.h>
#include <experimental/mdspan>
#include <cassert>
#include <vector>
#include <array>

namespace Scilib {

// Non-owing view of matrix diagonal.
template <class T>
class diag {
public:
    using value_type = T;
    using size_type = std::size_t;

    diag() = delete;

    diag(Matrix_view<T> m)
        : span{m.data(),
               {stdex::dextents<1>{m.extent(0)},
                std::array<std::size_t, 1>{m.stride(0) + 1}}}
    {
        static_assert(m.static_extent(0) == m.static_extent(1));
    }

    diag& operator=(const Vector_view<T> v)
    {
        assert(v.extent(0) == extent(0));
        for (size_type i = 0; i < v.extent(0); ++i) {
            span(i) = v(i);
        }
        return *this;
    }

    diag& operator=(const std::vector<T>& v)
    {
        assert(v.extent(0) == extent(0));
        for (size_type i = 0; i < v.size(); ++i) {
            span(i) = v(i);
        }
        return *this;
    }

    ~diag() = default;

    constexpr auto& operator()(std::size_t i) noexcept { return span(i); }
    constexpr auto& operator()(std::size_t i) const noexcept { return span(i); }

    std::size_t size() const noexcept { return span.size(); }

    std::size_t extent(std::size_t dim = 0) const noexcept
    {
        return span.extent(dim);
    }

    T* data() noexcept { return span.data(); }
    const T* data() const noexcept { return span.data(); }

    auto view() noexcept { return span; }
    const auto view() const noexcept { return span; }

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
