// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <experimental/mdspan>
#include <functional>
#include <scilib/mdarray_impl/copy.h>
#include <type_traits>
#include <vector>

namespace Sci {
namespace stdex = std::experimental;

namespace __Detail {

template <class Extents, class... Dims>
inline bool __check_bounds(const Extents& exts, Dims... dims)
{
    using size_type = std::size_t;

    std::vector<size_type> indexes{static_cast<size_type>(dims)...};
    bool result = true;
    for (size_type i = 0; i < indexes.size(); ++i) {
        if (!(indexes[i] < exts.extent(i))) {
            result = false;
        }
    }
    return result;
}

} // namespace __Detail

// Dense multidimensional array class for numerical computing using mdspan
// for views.
//
// Storage order can be either row-major (layout_right; default) or
// column-major (layout_left).
//
template <class T, class Extents, class Layout, class Allocator>
    requires Extents_has_rank<Extents>
class MDArray {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using extents_type = Extents;
    using layout_type = Layout;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using container_type = std::vector<value_type, Allocator>;
    using view_type = stdex::mdspan<value_type, extents_type, layout_type>;
    using const_view_type =
        stdex::mdspan<const value_type, extents_type, layout_type>;
    using difference_type = typename container_type::difference_type;
    using pointer = typename container_type::pointer;
    using const_pointer = typename container_type::const_pointer;
    using reference = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator =
        typename container_type::const_reverse_iterator;

    //--------------------------------------------------------------------------
    // Constructors, assignment and destructor

    constexpr MDArray() = default;

    template <class... Exts>
    constexpr explicit MDArray(Exts... exts)
        : map(extents_type(static_cast<size_type>(exts)...)),
          ctr(map.required_span_size())
    {
        static_assert(sizeof...(exts) == extents_type::rank());
    }

    template <class... Exts>
    constexpr MDArray(const std::vector<T>& m, Exts... exts)
        : map(extents_type(static_cast<size_type>(exts)...)), ctr(m)
    {
        static_assert(sizeof...(exts) == extents_type::rank());
        assert(m.size() == map.required_span_size());
    }

    template <std::size_t N>
    constexpr MDArray(const std::array<T, N>& a,
                      const std::array<size_type, extents_type::rank()>& exts)
        : map(exts), ctr(a.begin(), a.end())
    {
        assert(N == std::accumulate(exts.begin(), exts.end(), size_type{1},
                                    std::multiplies<size_type>()));
    }

    constexpr MDArray(MDArray&&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    constexpr MDArray(const MDArray& m) = default;

    template <class T_m, class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray(stdex::mdspan<T_m, Extents_m, Layout_m, Accessor_m> m)
        : map(m.extents()), ctr(m.size())
    {
        static_assert(m.rank() == extents_type::rank());
        static_assert(m.rank() <= 7);
        copy(m, view());
    }

    constexpr MDArray& operator=(const MDArray& m) = default;

    template <class T_m, class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray&
    operator=(stdex::mdspan<T_m, Extents_m, Layout_m, Accessor_m> m)
    {
        static_assert(m.rank() == extents_type::rank());
        static_assert(m.rank() <= 7);

        map = mapping_type(m.extents());
        ctr = container_type(m.size());

        copy(m, view());
        return *this;
    }

    ~MDArray() = default;

    //--------------------------------------------------------------------------
    // Mapping multidimensional index to access element

    template <class... Indices>
    constexpr reference operator()(Indices... indices) noexcept
    {
        static_assert(sizeof...(indices) == extents_type::rank());
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }

    template <class... Indices>
    constexpr const_reference operator()(Indices... indices) const noexcept
    {
        static_assert(sizeof...(indices) == extents_type::rank());
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }

    //--------------------------------------------------------------------------
    // Iterators

    constexpr iterator begin() noexcept { return ctr.begin(); }

    constexpr const_iterator begin() const noexcept { return ctr.begin(); }
    constexpr const_iterator cbegin() const noexcept { return ctr.cbegin(); }

    constexpr reverse_iterator rbegin() noexcept { return ctr.rbegin(); }

    constexpr const_reverse_iterator rbegin() const noexcept
    {
        return ctr.rbegin();
    }

    constexpr const_reverse_iterator crbegin() const noexcept
    {
        return ctr.crbegin();
    }

    constexpr iterator end() noexcept { return ctr.end(); }

    constexpr const_iterator end() const noexcept { return ctr.end(); }
    constexpr const_iterator cend() const noexcept { return ctr.cend(); }

    constexpr reverse_iterator rend() noexcept { return ctr.rend(); }

    constexpr const_reverse_iterator rend() const noexcept
    {
        return ctr.rend();
    }

    constexpr const_reverse_iterator crend() const noexcept
    {
        return ctr.crend();
    }

    //--------------------------------------------------------------------------
    // Access underlying data

    constexpr pointer data() noexcept { return ctr.data(); }
    constexpr const_pointer data() const noexcept { return ctr.data(); }

    constexpr container_type& container() noexcept { return ctr; }
    constexpr const container_type& container() const noexcept { return ctr; }

    //--------------------------------------------------------------------------
    // Return view of data

    constexpr view_type view() noexcept
    {
        return view_type(ctr.data(), map.extents());
    }

    constexpr const_view_type view() const noexcept
    {
        return const_view_type(ctr.data(), map.extents());
    }

    //--------------------------------------------------------------------------
    // Observers of the multidimensional index space

    constexpr static size_type rank() noexcept { return extents_type::rank(); }

    constexpr bool empty() const noexcept { return ctr.empty(); }

    constexpr size_type size() const noexcept { return ctr.size(); }

    constexpr difference_type ssize() const noexcept
    {
        return static_cast<difference_type>(ctr.size());
    }

    constexpr size_type max_size() const noexcept { return ctr.max_size(); }

    constexpr extents_type extents() const noexcept { return map.extents(); }

    constexpr size_type extent(size_type dim) const noexcept
    {
        assert(dim >= 0 && dim < extents_type::rank());
        return map.extents().extent(dim);
    }

    // Signed extent.
    constexpr difference_type sextent(size_type dim) const noexcept
    {
        assert(dim >= 0 && dim < extents_type::rank());
        return static_cast<difference_type>(map.extents().extent(dim));
    }

    //--------------------------------------------------------------------------
    // Observers of the mapping

    constexpr mapping_type mapping() const noexcept { return map; }

    constexpr size_type stride(size_type dim) const noexcept
    {
        return map.stride(dim);
    }

    //--------------------------------------------------------------------------
    // Modifiers

    template <class... Exts>
    constexpr void resize(Exts... exts) noexcept
    {
        static_assert(sizeof...(exts) == extents_type::rank());
        map = mapping_type(extents_type(static_cast<size_type>(exts)...));
        ctr = container_type(map.required_span_size());
    }

    constexpr void swap(MDArray& m) noexcept
    {
        std::swap(map, m.map);
        std::swap(ctr, m.ctr);
    }

    template <class F>
    constexpr MDArray& apply(F f) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(ctr[i]);
        }
        return *this;
    }

    template <class F, class U>
    constexpr MDArray& apply(F f, const U& val) noexcept
    {
        for (size_type i = 0; i < size(); ++i) {
            f(ctr[i], val);
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
    mapping_type map;
    container_type ctr;
};

} // namespace Sci

#endif // SCILIB_MDARRAY_BITS_H
