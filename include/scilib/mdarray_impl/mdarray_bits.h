// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#include <scilib/mdarray_impl/copy.h>
#include <experimental/mdspan>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <cassert>

namespace Scilib {
namespace stdex = std::experimental;

namespace __Detail {

template <class... Exts>
constexpr stdex::extents<>::size_type __compute_size(Exts... exts)
{
    using size_type = stdex::extents<>::size_type;

    std::vector<size_type> storage{static_cast<size_type>(exts)...};
    return std::accumulate(storage.begin(), storage.end(), size_type{1},
                           std::multiplies<size_type>());
}

template <class Extents, class... Dims>
constexpr bool __check_bounds(const Extents& exts, Dims... dims)
{
    using size_type = stdex::extents<>::size_type;

    std::vector<size_type> indexes{static_cast<size_type>(dims)...};
    bool result = true;
    for (size_type i = 0; i < indexes.size(); ++i) {
        if (indexes[i] < 0 || indexes[i] >= exts.extent(i)) {
            result = false;
        }
    }
    return result;
}

} // namespace __Detail

// Dense multidimensional array class with row-major storage order and using
// mdspan for views.
//
// clang-format off
template <class T, class Extents>
    requires Extents_has_rank<Extents> 
class MDArray {
public:
    // clang-format on
    using value_type = T;
    using size_type = stdex::extents<>::size_type;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reverse_iterator = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator =
        typename std::vector<T>::const_reverse_iterator;

    constexpr static size_type N_dim = Extents::rank();

    constexpr MDArray() = default;

    template <class... Exts>
    constexpr explicit MDArray(Exts... exts)
        : storage(__Detail::__compute_size(static_cast<size_type>(exts)...)),
          span(storage.data(),
               std::array<size_type, N_dim>{static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == N_dim);
    }

    template <class... Exts>
    constexpr MDArray(const std::vector<T>& m, Exts... exts)
        : storage(m),
          span(storage.data(),
               std::array<size_type, N_dim>{static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == N_dim);
        assert(m.size() ==
               __Detail::__compute_size(static_cast<size_type>(exts)...));
    }

    template <std::size_t N>
    constexpr MDArray(const std::array<T, N>& a,
                      const std::array<size_type, N_dim>& exts)
        : storage(a.begin(), a.end()), span(storage.data(), exts)
    {
        assert(N == std::accumulate(exts.begin(), exts.end(), std::size_t{1},
                                    std::multiplies<std::size_t>()));
    }

    MDArray(MDArray&&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    constexpr MDArray(const MDArray& m)
        : storage(m.storage), span(storage.data(), m.view().extents())
    {
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
        : storage(m.size()), span(storage.data(), m.extents())
    {
        static_assert(m.rank() == N_dim);
        static_assert(m.rank() <= 4);
        copy(m, span);
    }

    constexpr MDArray& operator=(const MDArray& m)
    {
        storage = m.storage;
        span = stdex::mdspan<T, Extents>(storage.data(), m.view().extents());
        return *this;
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray&
    operator=(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
    {
        static_assert(m.rank() == N_dim);
        static_assert(m.rank() <= 4);

        storage = std::vector<T>(m.size());
        span = stdex::mdspan<T, Extents>(storage.data(), m.extents());

        copy(m, span);
        return *this;
    }

    constexpr ~MDArray() = default;

    template <class... Indices>
    constexpr reference operator()(Indices... indices) noexcept
    {
        static_assert(sizeof...(indices) == N_dim);
        assert(__Detail::__check_bounds(span.extents(), indices...));
        return span(indices...);
    }

    template <class... Indices>
    constexpr const_reference operator()(Indices... indices) const noexcept
    {
        static_assert(sizeof...(indices) == N_dim);
        assert(__Detail::__check_bounds(span.extents(), indices...));
        return span(indices...);
    }

    constexpr iterator begin() noexcept { return storage.begin(); }
    constexpr const_iterator begin() const noexcept { return storage.begin(); }
    constexpr const_iterator cbegin() const noexcept
    {
        return storage.cbegin();
    }

    constexpr reverse_iterator rbegin() noexcept { return storage.rbegin(); }
    constexpr const_reverse_iterator rbegin() const noexcept
    {
        return storage.rbegin();
    }
    constexpr const_reverse_iterator crbegin() const noexcept
    {
        return storage.crbegin();
    }

    constexpr iterator end() noexcept { return storage.end(); }
    constexpr const_iterator end() const noexcept { return storage.end(); }
    constexpr const_iterator cend() const noexcept { return storage.cend(); }

    constexpr reverse_iterator rend() noexcept { return storage.rend(); }
    constexpr const_reverse_iterator rend() const noexcept
    {
        return storage.rend();
    }
    constexpr const_reverse_iterator crend() const noexcept
    {
        return storage.crend();
    }

    constexpr T* data() noexcept { return storage.data(); }
    constexpr const T* data() const noexcept { return storage.data(); }

    // Note: view() does not propagate const.
    constexpr auto view() const noexcept
    {
        return stdex::mdspan<T, Extents>(span);
    }

    constexpr size_type rank() const noexcept { return span.rank(); }

    constexpr bool empty() const noexcept { return storage.empty(); }

    constexpr size_type size() const noexcept { return storage.size(); }

    constexpr size_type max_size() const noexcept { return storage.max_size(); }

    constexpr size_type extent(size_type dim) const noexcept
    {
        assert(dim >= 0 && dim < N_dim);
        return span.extent(dim);
    }

    template <class... Exts>
    constexpr void resize(Exts... exts) noexcept
    {
        static_assert(sizeof...(exts) == N_dim);
        storage = std::vector<T>(
            __Detail::__compute_size(static_cast<size_type>(exts)...));
        span = stdex::mdspan<T, Extents>(
            storage.data(),
            std::array<size_type, N_dim>{static_cast<size_type>(exts)...});
    }

    constexpr void swap(MDArray& m) noexcept
    {
        std::swap(storage, m.storage);
        std::swap(span, m.span);
    }

    template <class F>
    constexpr MDArray& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(storage[i]);
        }
        return *this;
    }

    template <class F, class U>
    constexpr MDArray& apply(F f, const U& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(storage[i], val);
        }
        return *this;
    }

    template <class F>
    constexpr MDArray& apply(const MDArray& m, F f) noexcept
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

    constexpr MDArray& operator=(const T& value) noexcept
    {
        return apply([&](T& a) { a = value; });
    }

    constexpr MDArray& operator+=(const T& value) noexcept
    {
        return apply([&](T& a) { a += value; });
    }

    constexpr MDArray& operator-=(const T& value) noexcept
    {
        return apply([&](T& a) { a -= value; });
    }

    constexpr MDArray& operator*=(const T& value) noexcept
    {
        return apply([&](T& a) { a *= value; });
    }

    constexpr MDArray& operator/=(const T& value) noexcept
    {
        return apply([&](T& a) { a /= value; });
    }

    constexpr MDArray& operator%=(const T& value) noexcept
    {
        return apply([&](T& a) { a %= value; });
    }

    constexpr MDArray& operator+=(const MDArray& m) noexcept
    {
        return apply(m, [](T& a, const T& b) { a += b; });
    }

    constexpr MDArray& operator-=(const MDArray& m) noexcept
    {
        return apply(m, [](T& a, const T& b) { a -= b; });
    }

private:
    std::vector<T> storage;
    stdex::mdspan<T, Extents> span;
};

} // namespace Scilib

#endif // SCILIB_MDARRAY_BITS_H
