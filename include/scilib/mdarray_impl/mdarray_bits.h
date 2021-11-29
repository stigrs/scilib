// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267) // caused by experimental/mdspan
#endif

#include <scilib/mdarray_impl/type_aliases.h>
#include <experimental/mdspan>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <cassert>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace Scilib {
namespace stdex = std::experimental;

namespace __detail {

template <class... Exts>
inline std::size_t __compute_size(Exts... exts)
{
    std::vector<std::size_t> storage{static_cast<std::size_t>(exts)...};
    constexpr std::size_t one = 1;
    return std::accumulate(storage.begin(), storage.end(), one,
                           std::multiplies<std::size_t>());
}

} // namespace __detail

// Dense multidimensional array class with row-major storage order and using
// mdspan for views.
template <class T, std::size_t N, class Extents>
class MDArray {
public:
    using value_type = T;
    using size_type = stdex::extents<>::size_type;
    using difference_type = std::ptrdiff_t;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    MDArray() = default;

    template <class... Exts>
    explicit MDArray(Exts... exts)
        : storage(__detail::__compute_size(static_cast<size_type>(exts)...)),
          span(storage.data(),
               std::array<size_type, N>{static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == N);
    }

    template <class... Exts>
    MDArray(const std::vector<T>& m, Exts... exts)
        : storage(m),
          span(storage.data(),
               std::array<size_type, N>{static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == N);
    }

    MDArray(T* p, const std::array<size_type, N>& exts)
        : storage(p, p + exts.size()), span(storage.data(), exts)
    {
        static_assert(exts.size() == N);
    }

    MDArray(MDArray&&) = default;
    MDArray& operator=(MDArray&&) = default;

    MDArray(const MDArray& m)
        : storage(m.storage), span(storage.data(), m.view().extents())
    {
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    MDArray(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
        : storage(m.size())
    {
        static_assert(m.rank() <= 2);

        if constexpr (m.rank() == 1) {
            span = stdex::mdspan<T, Extents>(
                storage.data(), std::array<size_type, N>{m.extent(0)});
            for (size_type i = 0; i < m.extent(0); ++i) {
                storage[i] = m(i);
            }
        }
        else if constexpr (m.rank() == 2) {
            span = stdex::mdspan<T, Extents>(
                storage.data(),
                std::array<size_type, N>{m.extent(0), m.extent(1)});
            for (size_type i = 0; i < m.extent(0); ++i) {
                for (size_type j = 0; j < m.extent(1); ++j) {
                    storage[i * view().stride(0) + j * view().stride(1)] =
                        m(i, j);
                }
            }
        }
    }

    MDArray& operator=(const MDArray& m)
    {
        storage = m.storage;
        span = stdex::mdspan<T, Extents>(storage.data(), m.view().extents());
        return *this;
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    MDArray& operator=(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
    {
        static_assert(m.rank() <= 2);

        storage = std::vector<T>(m.size());
        if constexpr (m.rank() == 1) {
            span = stdex::mdspan<T, Extents>(
                storage.data(), std::array<size_type, N>{m.extent(0)});
            for (size_type i = 0; i < m.extent(0); ++i) {
                storage[i] = m(i);
            }
        }
        else if constexpr (m.rank() == 2) {
            span = stdex::mdspan<T, Extents>(
                storage.data(),
                std::array<size_type, N>{m.extent(0), m.extent(1)});
            for (size_type i = 0; i < m.extent(0); ++i) {
                for (size_type j = 0; j < m.extent(1); ++j) {
                    storage[i * view().stride(0) + j * view().stride(1)] =
                        m(i, j);
                }
            }
        }
        return *this;
    }

    MDArray& operator=(const std::vector<T>& v)
    {
        static_assert(N == 1);

        storage = std::vector<T>(v.size());
        span = stdex::mdspan<T, Extents>(storage.data(),
                                         std::array<size_type, N>{v.size()});
        for (size_type i = 0; i < v.size(); ++i) {
            storage[i] = v[i];
        }
        return *this;
    }

    template <class... Indices>
    constexpr auto& operator()(Indices... indices) noexcept
    {
        static_assert(sizeof...(indices) == N);
        return span(indices...);
    }

    template <class... Indices>
    constexpr const auto& operator()(Indices... indices) const noexcept
    {
        static_assert(sizeof...(indices) == N);
        return span(indices...);
    }

    constexpr iterator begin() noexcept { return storage.begin(); }
    constexpr iterator end() noexcept { return storage.end(); }

    constexpr const_iterator begin() const noexcept { return storage.begin(); }
    constexpr const_iterator end() const noexcept { return storage.end(); }

    constexpr T* data() noexcept { return storage.data(); }
    constexpr const T* data() const noexcept { return storage.data(); }

    constexpr auto view() noexcept { return span; }
    constexpr const auto view() const noexcept { return span; }

    constexpr bool empty() const noexcept { return storage.empty(); }
    constexpr auto size() const noexcept { return span.size(); }

    constexpr size_type rank() const noexcept
    {
        return static_cast<size_type>(span.rank());
    }

    constexpr auto extent(size_type dim) const noexcept
    {
        return span.extent(dim);
    }

    template <class... Exts>
    void resize(Exts... exts) noexcept
    {
        static_assert(sizeof...(exts) == N);
        storage = std::vector<T>(
            __detail::__compute_size(static_cast<size_type>(exts)...));
        span = stdex::mdspan<T, Extents>(
            storage.data(),
            std::array<size_type, N>{static_cast<size_type>(exts)...});
    }

    void swap(MDArray& m) noexcept
    {
        std::swap(storage, m.storage);
        std::swap(span, m.span);
    }

    // Apply f(x) for every element x.
    template <class F>
    MDArray& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(storage[i]);
        }
        return *this;
    }

    // Apply f(x, val) for every element x.
    template <class F>
    MDArray& apply(F f, const T& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(storage[i], val);
        }
        return *this;
    }

    template <class F>
    MDArray& apply(const MDArray& m, F f) noexcept
    {
        assert(view().extents() == m.view().extents());

        auto i = begin();
        auto j = m.begin();
        while (i != end()) {
            f(*i, *j);
            ++i;
            ++j;
        }
        return *this;
    }

    MDArray& operator=(const T& value) noexcept
    {
        return apply([&](T& a) { a = value; });
    }

    MDArray& operator+=(const T& value) noexcept
    {
        return apply([&](T& a) { a += value; });
    }

    MDArray& operator-=(const T& value) noexcept
    {
        return apply([&](T& a) { a -= value; });
    }

    MDArray& operator*=(const T& value) noexcept
    {
        return apply([&](T& a) { a *= value; });
    }

    MDArray& operator/=(const T& value) noexcept
    {
        return apply([&](T& a) { a /= value; });
    }

    MDArray& operator%=(const T& value) noexcept
    {
        return apply([&](T& a) { a %= value; });
    }

    MDArray& operator+=(const MDArray& m) noexcept
    {
        // static_assert(view().extents() == m.view().extents());
        return apply(m, [](T& a, const T& b) { a += b; });
    }

    MDArray& operator-=(const MDArray& m) noexcept
    {
        // static_assert(view().extents() == m.view().extents());
        return apply(m, [](T& a, const T& b) { a -= b; });
    }

private:
    std::vector<T> storage;
    stdex::mdspan<T, Extents> span;
};

template <class T>
using Vector = MDArray<T, 1, stdex::extents<stdex::dynamic_extent>>;

template <class T>
using Matrix =
    MDArray<T, 2, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>;

} // namespace Scilib

#endif // SCILIB_MDARRAY_BITS_H
