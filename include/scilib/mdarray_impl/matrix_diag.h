// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray_impl/matrix.h>
#include <scilib/mdarray_impl/vector.h>
#include <cassert>
#include <vector>
#include <array>

namespace Scilib {

// Non-owing view of matrix diagonal.
template <typename T>
class diag {
public:
    using value_type = T;
    using size_type = std::size_t;

    diag() = delete;

    diag(Matrix_view<T>& m)
        : elems(m.data()),
          ext{m.extent(0), m.extent(1)},
          sz{m.size()},
          st{m.stride(0) + 1}
    {
        assert(m.extent(0) == m.extent(1));
    }

    diag& operator=(const Vector_view<T>& v)
    {
        assert(v.extent(0) == extent(0));
        for (size_type i = 0; i < v.extent(0); ++i) {
            elems[offset(i)] = v(i);
        }
        return *this;
    }

    diag& operator=(const std::vector<T>& v)
    {
        assert(v.extent(0) == extent(0));
        for (size_type i = 0; i < v.size(); ++i) {
            elems[offset(i)] = v[i];
        }
        return *this;
    }

    ~diag() = default;

    auto& operator()(std::size_t i) { return elems[i * st]; }
    const auto& operator()(std::size_t i) const { return elems[i * st]; }

    std::size_t size() const { return sz; }
    std::size_t extent(std::size_t dim = 0) const { return ext[dim]; }
    std::size_t stride(std::size_t dim = 0) const { return st; }

    T* data() { return elems; }
    const T* data() const { return elems; }

    // Apply f(x) for every element x.
    template <class F>
    diag& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[offset(i)]);
        }
        return *this;
    }

    // Apply f(x, val) for every element x.
    template <class F>
    diag& apply(F f, const T& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[offset(i)], val);
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
    size_type offset(size_type i) const { return i * st; }

    T* elems;
    std::array<std::size_t, 2> ext;
    std::size_t sz;
    std::size_t st;
};

} // namespace Scilib
