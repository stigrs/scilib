// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray_impl/type_aliases.h>
#include <experimental/mdspan>
#include <vector>
#include <initializer_list>
#include <utility>
#include <cassert>

namespace Scilib {
namespace stdex = std::experimental;

// Dense matrix class with row-major storage order and using mdspan for views.
template <class T>
class Matrix {
public:
    using value_type = T;
    using size_type = typename Matrix_view<T>::size_type;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    Matrix() = default;

    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;

    Matrix(const Matrix& m)
        : elems(m.size()), span(elems.data(), m.rows(), m.cols())
    {
        size_type i = 0;
        for (const auto& mi : m) {
            elems[i] = mi;
            ++i;
        }
    }

    Matrix(size_type nr, size_type nc)
        : elems(nr * nc), span(elems.data(), nr, nc)
    {
    }

    Matrix(size_type nr, size_type nc, const T& val)
        : elems(nr * nc, val), span(elems.data(), nr, nc)
    {
    }

    Matrix(T* p, size_type nr, size_type nc)
        : elems(p, p + nr * nc), span(elems.data(), nr, nc)
    {
    }

    Matrix(const std::vector<T>& v, size_type nr, size_type nc)
        : elems(v), span(elems.data(), nr, nc)
    {
    }

    template <stdex::extents<>::size_type nrows,
              stdex::extents<>::size_type ncols,
              class Layout,
              class Accessor>
    Matrix(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m)
        : elems(m.size()), span(elems.data(), m.extent(0), m.extent(1))
    {
        for (size_type i = 0; i < m.extent(0); ++i) {
            for (size_type j = 0; j < m.extent(1); ++j) {
                elems[i * view().stride(0) + j * view().stride(1)] = m(i, j);
            }
        }
    }

    template <stdex::extents<>::size_type nrows,
              stdex::extents<>::size_type ncols,
              class Layout,
              class Accessor>
    Matrix& operator=(
        stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m)
        : elems(m.size()), span(elems.data(), m.extent(0), m.extent(1))
    {
        elems = std::vector<T>(m.size());
        span = Matrix_view<T>(elems.data(), m.rows(), m.cols());

        for (size_type i = 0; i < m.extent(0); ++i) {
            for (size_type j = 0; j < m.extent(1); ++j) {
                elems[i * view().stride(0) + j * view().stride(1)] = m(i, j);
            }
        }
        return *this;
    }

    ~Matrix() = default;

    constexpr auto& operator()(size_type i, size_type j) noexcept
    {
        return span(i, j);
    }

    constexpr auto& operator()(size_type i, size_type j) const noexcept
    {
        return span(i, j);
    }

    iterator begin() noexcept { return elems.begin(); }
    iterator end() noexcept { return elems.end(); }

    const_iterator begin() const noexcept { return elems.begin(); }
    const_iterator end() const noexcept { return elems.end(); }

    T* data() noexcept { return elems.data(); }
    const T* data() const noexcept { return elems.data(); }

    auto view() noexcept { return span; }
    const auto view() const noexcept { return span; }

    bool empty() const noexcept { return elems.empty(); }
    auto size() const noexcept { return span.size(); }

    auto rows() const noexcept { return span.extent(0); }
    auto cols() const noexcept { return span.extent(1); }

    auto extent(size_type dim) const noexcept
    {
        assert(dim == 0 || dim == 1);
        return span.extent(dim);
    }

    auto stride(size_type dim) const noexcept
    {
        assert(dim == 0 || dim == 1);
        return span.stride(dim);
    }

    auto row(size_type i) noexcept
    {
        return stdex::submdspan(span, i, stdex::full_extent);
    }

    const auto row(size_type i) const noexcept
    {
        return stdex::submdspan(span, i, stdex::full_extent);
    }

    auto column(size_type j) noexcept
    {
        return stdex::submdspan(span, stdex::full_extent, j);
    }

    const auto column(size_type j) const noexcept
    {
        return stdex::submdspan(span, stdex::full_extent, j);
    }

    void resize(size_type nr, size_type nc)
    {
        elems = std::vector<T>(nr * nc);
        span = Matrix_view<T>(elems.data(), nr, nc);
    }

    void swap(Matrix& m) noexcept
    {
        std::swap(elems, m.elems);
        std::swap(span, m.span);
    }

    // Apply f(x) for every element x.
    template <class F>
    Matrix& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[i]);
        }
        return *this;
    }

    // Apply f(x, val) for every element x.
    template <class F>
    Matrix& apply(F f, const T& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(elems[i], val);
        }
        return *this;
    }

    template <class F>
    Matrix& apply(const Matrix& m, F f) noexcept
    {
        assert(size() == m.size());
        auto i = begin();
        auto j = m.begin();
        while (i != end()) {
            f(*i, *j);
            ++i;
            ++j;
        }
        return *this;
    }

    Matrix& operator=(const T& value) noexcept
    {
        return apply([&](T& a) { a = value; });
    }

    Matrix& operator+=(const T& value) noexcept
    {
        return apply([&](T& a) { a += value; });
    }

    Matrix& operator-=(const T& value) noexcept
    {
        return apply([&](T& a) { a -= value; });
    }

    Matrix& operator*=(const T& value) noexcept
    {
        return apply([&](T& a) { a *= value; });
    }

    Matrix& operator/=(const T& value) noexcept
    {
        return apply([&](T& a) { a /= value; });
    }

    Matrix& operator%=(const T& value) noexcept
    {
        return apply([&](T& a) { a %= value; });
    }

    Matrix& operator+=(const Matrix& m) noexcept
    {
        assert(size() == m.size());
        return apply(m, [](T& a, const T& b) { a += b; });
    }

    Matrix& operator-=(const Matrix& m) noexcept
    {
        assert(size() == m.size());
        return apply(m, [](T& a, const T& b) { a -= b; });
    }

private:
    std::vector<T> elems;
    Matrix_view<T> span;
};

} // namespace Scilib
