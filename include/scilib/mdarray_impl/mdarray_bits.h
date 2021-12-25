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
inline stdex::extents<>::size_type __compute_size(Exts... exts)
{
    using size_type = stdex::extents<>::size_type;

    std::vector<size_type> extents{static_cast<size_type>(exts)...};
    return std::accumulate(extents.begin(), extents.end(), size_type{1},
                           std::multiplies<size_type>());
}

template <class Extents, class... Dims>
inline bool __check_bounds(const Extents& exts, Dims... dims)
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

// Dense multidimensional array class for numerical computing with row-major
// storage order and using mdspan for views.
//
// clang-format off
template <class T, class Extents, class Layout, class ContainerType>
    requires Extents_has_rank<Extents> 
class MDArray {
public:
    // clang-format on
    using value_type = T;
    using layout_type = Layout;
    using container_type = ContainerType;
    using view_type = stdex::mdspan<T, Extents, layout_type>;
    using const_view_type = stdex::mdspan<const T, Extents, layout_type>;
    using size_type = stdex::extents<>::size_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator =
        typename container_type::const_reverse_iterator;

    constexpr MDArray() = default;

    template <class... Exts>
    constexpr explicit MDArray(Exts... exts)
        : c_(__Detail::__compute_size(static_cast<size_type>(exts)...)),
          v_(c_.data(),
             std::array<size_type, Extents::rank()>{
                 static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == Extents::rank());
    }

    template <class... Exts>
    constexpr MDArray(const std::vector<T>& m, Exts... exts)
        : c_(m),
          v_(c_.data(),
             std::array<size_type, Extents::rank()>{
                 static_cast<size_type>(exts)...})
    {
        static_assert(sizeof...(exts) == Extents::rank());
        assert(m.size() ==
               __Detail::__compute_size(static_cast<size_type>(exts)...));
    }

    template <std::size_t N>
    constexpr MDArray(const std::array<T, N>& a,
                      const std::array<size_type, Extents::rank()>& exts)
        : c_(a.begin(), a.end()), v_(c_.data(), exts)
    {
        assert(N == std::accumulate(exts.begin(), exts.end(), size_type{1},
                                    std::multiplies<size_type>()));
    }

    constexpr MDArray(MDArray&&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    constexpr MDArray(const MDArray& m)
        : c_(m.c_), v_(c_.data(), m.view().extents())
    {
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
        : c_(m.size()), v_(c_.data(), m.extents())
    {
        static_assert(m.rank() == Extents::rank());
        static_assert(m.rank() <= 4);
        copy(m, v_);
    }

    constexpr MDArray& operator=(const MDArray& m)
    {
        c_ = m.c_;
        v_ = view_type(c_.data(), m.view().extents());
        return *this;
    }

    template <class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray&
    operator=(stdex::mdspan<T, Extents_m, Layout_m, Accessor_m> m)
    {
        static_assert(m.rank() == Extents::rank());
        static_assert(m.rank() <= 4);

        c_ = container_type(m.size());
        v_ = view_type(c_.data(), m.extents());

        copy(m, v_);
        return *this;
    }

    ~MDArray() = default;

    constexpr reference operator[](size_type idx) noexcept
    {
        assert(idx >= 0 && idx < size());
        return c_[idx];
    }

    constexpr const_reference operator[](size_type idx) const noexcept
    {
        assert(idx >= 0 && idx < size());
        return c_[idx];
    }

    template <class... Indices>
    constexpr reference operator()(Indices... indices) noexcept
    {
        static_assert(sizeof...(indices) == Extents::rank());
        assert(__Detail::__check_bounds(v_.extents(), indices...));
        return v_(indices...);
    }

    template <class... Indices>
    constexpr const_reference operator()(Indices... indices) const noexcept
    {
        static_assert(sizeof...(indices) == Extents::rank());
        assert(__Detail::__check_bounds(v_.extents(), indices...));
        return v_(indices...);
    }

    constexpr iterator begin() noexcept { return c_.begin(); }
    constexpr const_iterator begin() const noexcept { return c_.begin(); }
    constexpr const_iterator cbegin() const noexcept { return c_.cbegin(); }

    constexpr reverse_iterator rbegin() noexcept { return c_.rbegin(); }
    constexpr const_reverse_iterator rbegin() const noexcept
    {
        return c_.rbegin();
    }
    constexpr const_reverse_iterator crbegin() const noexcept
    {
        return c_.crbegin();
    }

    constexpr iterator end() noexcept { return c_.end(); }
    constexpr const_iterator end() const noexcept { return c_.end(); }
    constexpr const_iterator cend() const noexcept { return c_.cend(); }

    constexpr reverse_iterator rend() noexcept { return c_.rend(); }
    constexpr const_reverse_iterator rend() const noexcept { return c_.rend(); }
    constexpr const_reverse_iterator crend() const noexcept
    {
        return c_.crend();
    }

    constexpr pointer data() noexcept { return c_.data(); }
    constexpr const_pointer data() const noexcept { return c_.data(); }

    constexpr view_type view() noexcept { return view_type(v_); }
    constexpr const_view_type view() const noexcept
    {
        return const_view_type(v_);
    }

    constexpr static size_type rank() noexcept { return Extents::rank(); }

    constexpr bool empty() const noexcept { return c_.empty(); }

    constexpr size_type size() const noexcept { return c_.size(); }

    constexpr size_type max_size() const noexcept { return c_.max_size(); }

    constexpr size_type extent(size_type dim) const noexcept
    {
        assert(dim >= 0 && dim < Extents::rank());
        return v_.extent(dim);
    }

    template <class... Exts>
    constexpr void resize(Exts... exts) noexcept
    {
        static_assert(sizeof...(exts) == Extents::rank());
        c_ = container_type(
            __Detail::__compute_size(static_cast<size_type>(exts)...));
        v_ = view_type(c_.data(), std::array<size_type, Extents::rank()>{
                                      static_cast<size_type>(exts)...});
    }

    constexpr void swap(MDArray& m) noexcept
    {
        std::swap(c_, m.c_);
        std::swap(v_, m.v_);
    }

    template <class F>
    constexpr MDArray& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(c_[i]);
        }
        return *this;
    }

    template <class F, class U>
    constexpr MDArray& apply(F f, const U& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(c_[i], val);
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
    container_type c_;
    view_type v_;
};

} // namespace Scilib

#endif // SCILIB_MDARRAY_BITS_H
