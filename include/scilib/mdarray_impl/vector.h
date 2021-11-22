// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <experimental/mdspan>
#include <vector>
#include <utility>
#include <cassert>

namespace Scilib {

namespace stdex = std::experimental;

template <typename T>
using Vector_view = stdex::mdspan<T, stdex::extents<stdex::dynamic_extent>>;

// Dense vector class using mdspan for views.
template <class T>
class Vector {
public:
    using value_type = T;
    using size_type = typename Vector_view<T>::size_type;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    Vector() = default;

    Vector(Vector&&) = default;
    Vector& operator=(Vector&&) = default;

    Vector(const Vector& v) : elems(v.size()), span(elems.data(), elems.size())
    {
        for (size_type i = 0; i < size(); ++i) {
            elems[i] = v(i);
        }
    }

    Vector(size_type n) : elems(n), span(elems.data(), n) {}
    Vector(size_type n, const T& val) : elems(n, val), span(elems.data(), n) {}
    Vector(T* p, size_type n) : elems(p, p + n), span(elems.data(), n) {}
    Vector(const std::vector<T>& v) : elems(v), span(elems.data(), elems.size())
    {
    }

    Vector(const Vector_view<T>& v)
        : elems(v.size()), span(elems.data(), elems.size())
    {
        for (size_type i = 0; i < size(); ++i) {
            elems[i] = v(i);
        }
    }

    Vector& operator=(const Vector& v)
    {
        elems = std::vector<T>(v.size());
        span = Vector_view<T>(elems.data(), elems.size());

        for (size_type i = 0; i < size(); ++i) {
            elems[i] = v(i);
        }
        return *this;
    }

    Vector& operator=(const std::vector<T>& v)
    {
        elems = v;
        span = Vector_view<T>(elems.data(), elems.size());

        for (size_type i = 0; i < size(); ++i) {
            elems[i] = v[i];
        }
        return *this;
    }

    Vector& operator=(const Vector_view<T>& v)
    {
        elems.resize(v.size());
        span = Vector_view<T>(elems.data(), elems.size());

        for (size_type i = 0; i < size(); ++i) {
            elems[i] = v(i);
        }
        return *this;
    }

    ~Vector() = default;

    auto& operator()(size_type i) noexcept { return span(i); }
    const auto& operator()(size_type i) const noexcept { return span(i); }

    iterator begin() noexcept { return elems.begin(); }
    iterator end() noexcept { return elems.end(); }

    const_iterator begin() const noexcept { return elems.begin(); }
    const_iterator end() const noexcept { return elems.end(); }

    T* data() noexcept { return elems.data(); }
    const T* data() const noexcept { return elems.data(); }

    auto& view() noexcept { return span; }
    const auto& view() const noexcept { return span; }

    bool empty() const noexcept { return elems.empty(); }
    auto size() const noexcept { return span.size(); }

    void resize(size_type n)
    {
        elems = std::vector<T>(n);
        span = Vector_view<T>(elems.data(), elems.size());
    }

    void swap(Vector& v) noexcept
    {
        std::swap(elems, v.elems);
        std::swap(span, v.span);
    }

    // Apply f(x) for every element x.
    template <class F>
    Vector& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[i]);
        }
        return *this;
    }

    // Apply f(x, val) for every element x.
    template <class F>
    Vector& apply(F f, const T& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[i], val);
        }
        return *this;
    }

    template <class F>
    Vector& apply(const Vector& v, F f) noexcept
    {
        assert(size() == v.size());
        auto i = begin();
        auto j = v.begin();
        while (i != end()) {
            f(*i, *j);
            ++i;
            ++j;
        }
        return *this;
    }

    Vector& operator=(const T& value) noexcept
    {
        return apply([&](T& a) { a = value; });
    }

    Vector& operator+=(const T& value) noexcept
    {
        return apply([&](T& a) { a += value; });
    }

    Vector& operator-=(const T& value) noexcept
    {
        return apply([&](T& a) { a -= value; });
    }

    Vector& operator*=(const T& value) noexcept
    {
        return apply([&](T& a) { a *= value; });
    }

    Vector& operator/=(const T& value) noexcept
    {
        return apply([&](T& a) { a /= value; });
    }

    Vector& operator%=(const T& value) noexcept
    {
        return apply([&](T& a) { a %= value; });
    }

    Vector& operator+=(const Vector& v) noexcept
    {
        assert(size() == v.size());
        return apply(v, [](T& a, const T& b) { a += b; });
    }

    Vector& operator-=(const Vector& v) noexcept
    {
        assert(size() == v.size());
        return apply(v, [](T& a, const T& b) { a -= b; });
    }

    Vector& operator+=(const Vector_view& v) noexcept
    {
        assert(size() == v.size());
        return apply(v, [](T& a, const T& b) { a += b; });
    }

    Vector& operator-=(const Vector_view& v) noexcept
    {
        assert(size() == v.size());
        return apply(v, [](T& a, const T& b) { a -= b; });
    }

private:
    std::vector<T> elems;
    Vector_view<T> span;
};

} // namespace Scilib
