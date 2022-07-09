// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#include "support.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <gsl/gsl>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace Sci {
namespace stdex = std::experimental;

namespace __Detail {

template <class Extents, class... Dims>
inline bool __check_bounds(const Extents& exts, Dims... dims)
{
    using index_type = typename Extents::index_type;

    std::vector<index_type> indexes{static_cast<index_type>(dims)...};
    bool result = true;
    for (std::size_t i = 0; i < indexes.size(); ++i) {
        if (!(indexes[i] >= 0 && indexes[i] < exts.extent(i))) {
            result = false;
        }
    }
    return result;
}

template <class T, class Extents, class Layout, class Accessor>
inline Extents extents(stdex::mdspan<T, Extents, Layout, Accessor> m)
{
    // mdspan returns m.extents() by reference, need to make a copy
    using index_type = typename Extents::index_type;

    std::array<index_type, Extents::rank()> res;
    for (std::size_t i = 0; i < m.rank(); ++i) {
        res[i] = m.extent(i);
    }
    return res;
}

} // namespace __Detail

// Dense multidimensional array class for numerical computing using mdspan
// for views.
//
// Storage order can be either row-major (layout_right; default) or
// column-major (layout_left).
//
// The design is based on the proposed mdarray (P1684R2) that is specified
// in the following paper:
//   https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1684r2.html
//
template <class ElementType, class Extents, class LayoutPolicy, class Container>
    requires Extents_has_rank<Extents>
class MDArray {
public:
    using element_type = ElementType;
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using container_type = Container;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename extents_type::index_type;
    using pointer = typename container_type::pointer;
    using const_pointer = typename container_type::const_pointer;
    using reference = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    // [MDArray.ctors], MDArray constructors

    constexpr MDArray() requires(extents_type::rank_dynamic() != 0) = default;
    constexpr MDArray(const MDArray&) = default;
    constexpr MDArray(MDArray&&) = default;

    template <class... SizeTypes>
        requires((extents_type::rank() > 0 || extents_type::rank_dynamic() == 0) &&
                 std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 std::is_constructible_v<extents_type, SizeTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type>)
    constexpr explicit MDArray(SizeTypes... exts)
        : map(extents_type(exts...)),
          ctr(__Detail::Container_is_array<container_type>::construct(map.required_span_size()))
    {
    }

    constexpr MDArray(const extents_type& exts) requires(
        std::is_constructible_v<mapping_type, extents_type>)
        : map(exts),
          ctr(__Detail::Container_is_array<container_type>::construct(map.required_span_size()))
    {
    }

    constexpr MDArray(const mapping_type& m)
        : map(m),
          ctr(__Detail::Container_is_array<container_type>::construct(map.required_span_size()))
    {
    }

    // clang-format off
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 std::is_constructible_v<extents_type, SizeTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type>)
    // clang-format on
    constexpr explicit MDArray(const container_type& c, SizeTypes... exts)
        : map(extents_type(exts...)), ctr(c)
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    constexpr MDArray(const container_type& c, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type>)
        : map(exts), ctr(c)
    // clang-format on
    {
        Expects(map.required_span_size() == c.size());
    }

    constexpr MDArray(const container_type& c, const mapping_type& m) : map(m), ctr(c)
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 std::is_constructible_v<extents_type, SizeTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type>)
    // clang-format on
    constexpr explicit MDArray(container_type&& c, SizeTypes... exts)
        : map(extents_type(exts...)), ctr(std::move(c))
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    constexpr MDArray(container_type&& c, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type>)
        : map(exts), ctr(std::move(c))
    // clang-format on
    {
        Expects(map.required_span_size() == c.size());
    }

    constexpr MDArray(container_type&& c, const mapping_type& m) : map(m), ctr(std::move(c))
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 std::is_constructible_v<extents_type, SizeTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::initializer_list<element_type>>)
    // clang-format on
    constexpr explicit MDArray(std::initializer_list<element_type> init, SizeTypes... exts)
        : map(extents_type(exts...)), ctr(init)
    {
        Expects(map.required_span_size() == init.size());
    }

    // clang-format off
    constexpr MDArray(std::initializer_list<element_type> init, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::initializer_list<element_type>>)
        : map(exts), ctr(init)
    // clang-format on
    {
        Expects(map.required_span_size() == init.size());
    }

    // clang-format off
    constexpr MDArray(std::initializer_list<element_type> init, const mapping_type& m) 
        requires(std::is_constructible_v<container_type, std::initializer_list<element_type>>)
        : map(m), ctr(init)
    // clang-format on
    {
        Expects(map.required_span_size() == init.size());
    }

    // clang-format off
    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class Accessor>
        requires(std::is_convertible_v<element_type, OtherElementType> &&
                 std::is_constructible_v<extents_type, OtherExtents>)
    // clang-format on
    constexpr MDArray(
        stdex::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor> other)
        : map(extents_type(__Detail::extents(other))), ctr(other.size())
    {
        static_assert(other.rank() <= 7);
        for (std::size_t r = 0; r < other.rank(); ++r) {
            Expects(extents().static_extent(r) == stdex::dynamic_extent ||
                    extents().static_extent(r) == other.extent(r));
        }
        copy(other, view());
    }

    // [MDArray.ctors.alloc], MDArray constructors with allocators

    // clang-format off
    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    // clang-format on
    constexpr MDArray(const extents_type& exts, const Alloc& a)
        : map(exts), ctr(map.required_span_size(), a)
    {
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const mapping_type& m, const Alloc& a)
        : map(m), ctr(map.required_span_size(), a)
    {
    }

    // clang-format off
    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    // clang-format on
    constexpr MDArray(const container_type& c, const extents_type& exts, const Alloc& a)
        : map(exts), ctr(c, a)
    {
        Expects(map.required_span_size() == c.size());
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const container_type& c, const mapping_type& m, const Alloc& a)
        : map(m), ctr(c, a)
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    // clang-format on
    constexpr MDArray(container_type&& c, const extents_type& exts, const Alloc& a)
        : map(exts), ctr(c, a)
    {
        Expects(map.required_span_size() == c.size());
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(container_type&& c, const mapping_type& m, const Alloc& a) : map(m), ctr(c, a)
    {
        Expects(map.required_span_size() == c.size());
    }

    // clang-format off
    template <class Alloc>
        requires(
            std::is_constructible_v<mapping_type, extents_type> &&
            std::is_constructible_v<container_type, std::initializer_list<element_type>, Alloc>)
    // clang-format on
    constexpr MDArray(std::initializer_list<element_type> init,
                      const extents_type& exts,
                      const Alloc& a)
        : map(exts), ctr(init, a)
    {
        Expects(map.required_span_size() == init.size());
    }

    // clang-format off
    template <class Alloc>
        requires(
            std::is_constructible_v<container_type, std::initializer_list<element_type>, Alloc>)
    // clang-format on
    constexpr MDArray(std::initializer_list<element_type> init,
                      const mapping_type& m,
                      const Alloc& a)
        : map(m), ctr(init, a)
    {
        Expects(map.required_span_size() == init.size());
    }

    // clang-format off
    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class Accessor,
              class Alloc>
        requires(std::is_convertible_v<element_type, OtherElementType> &&
                 std::is_constructible_v<extents_type, OtherExtents> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    // clang-format on
    constexpr MDArray(
        stdex::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor> other,
        const Alloc& a)
        : map(extents_type(__Detail::extents(other))), ctr(other.size(), a)
    {
        static_assert(other.rank() <= 7);
        for (std::size_t r = 0; r < other.rank(); ++r) {
            Expects(extents().static_extent(r) == stdex::dynamic_extent ||
                    extents().static_extent(r) == other.extent(r));
        }
        copy(other, view());
    }

    // [MDArray.members], MDArray members

    constexpr const extents_type& extents() const noexcept { return map.extents(); }
    constexpr index_type extent(std::size_t r) const
    {
        Expects(r < extents_type::rank());
        return map.extents().extent(r);
    }

    constexpr std::size_t size() const noexcept { return ctr.size(); }

    constexpr pointer data() noexcept { return ctr.data(); }
    constexpr const_pointer data() const noexcept { return ctr.data(); }

    template <class OtherAccessorType>
    constexpr stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>
    view(const OtherAccessorType& a = stdex::default_accessor<element_type>)
    {
        return stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>(data(),
                                                                                         map, a);
    }

private:
    mapping_type map;
    container_type ctr;
};
#if 0
template <class T, class Extents, class Layout, class Allocator>
    requires Extents_has_rank<Extents>
class MDArray {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using extents_type = Extents;
    using layout_type = Layout;
    using index_type = typename extents_type::index_type;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using container_type = std::vector<value_type, Allocator>;
    using view_type = stdex::mdspan<value_type, extents_type, layout_type>;
    using const_view_type = stdex::mdspan<const value_type, extents_type, layout_type>;
    using difference_type = typename container_type::difference_type;
    using pointer = typename container_type::pointer;
    using const_pointer = typename container_type::const_pointer;
    using reference = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    //----------------------------------------------------------------------------------------------
    // Constructors, assignment and destructor

    constexpr MDArray() = default;

    template <class... Exts>
        requires(std::conjunction_v<std::is_convertible<Exts, index_type>...> &&
                 sizeof...(Exts) == extents_type::rank())
    constexpr explicit MDArray(Exts... exts)
        : map(extents_type(static_cast<index_type>(exts)...)), ctr(map.required_span_size())
    {
    }

    template <class... Exts>
        requires(std::conjunction_v<std::is_convertible<Exts, index_type>...> &&
                 sizeof...(Exts) == extents_type::rank())
    constexpr MDArray(const std::vector<T>& m, Exts... exts)
        : map(extents_type(static_cast<index_type>(exts)...)), ctr(m)
    {
        assert(m.size() == map.required_span_size());
    }

    template <std::size_t N>
    constexpr MDArray(const std::array<T, N>& a,
                      const std::array<index_type, extents_type::rank()>& exts)
        : map(exts), ctr(a.begin(), a.end())
    {
        assert(N == std::accumulate(exts.begin(), exts.end(), index_type{1},
                                    std::multiplies<index_type>()));
    }

    constexpr MDArray(MDArray&&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    constexpr MDArray(const MDArray& m) = default;

    template <class T_m, class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray(stdex::mdspan<T_m, Extents_m, Layout_m, Accessor_m> m)
        : map(extents_type(__Detail::extents(m))), ctr(m.size())
    {
        static_assert(m.rank() == extents_type::rank());
        static_assert(m.rank() <= 7);
        copy(m, view());
    }

    constexpr MDArray& operator=(const MDArray& m) = default;

    template <class T_m, class Extents_m, class Layout_m, class Accessor_m>
    constexpr MDArray& operator=(stdex::mdspan<T_m, Extents_m, Layout_m, Accessor_m> m)
    {
        static_assert(m.rank() == extents_type::rank());
        static_assert(m.rank() <= 7);

        map = mapping_type(extents_type(__Detail::extents(m)));
        ctr = container_type(m.size());

        copy(m, view());
        return *this;
    }

    ~MDArray() = default;

    //----------------------------------------------------------------------------------------------
    // Mapping multidimensional index to access element

#if MDSPAN_USE_PAREN_OPERATOR
    template <class... Indices>
        requires(std::conjunction_v<std::is_convertible<Indices, index_type>...> &&
                 sizeof...(Indices) == extents_type::rank())
    constexpr reference operator()(Indices... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }

    template <class... Indices>
        requires(std::conjunction_v<std::is_convertible<Indices, index_type>...> &&
                 sizeof...(Indices) == extents_type::rank())
    constexpr const_reference operator()(Indices... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }
#endif

#if MDSPAN_USE_BRACKET_OPERATOR
    template <class... Indices>
        requires(std::conjunction_v<std::is_convertible<Indices, index_type>...> &&
                 sizeof...(Indices) == extents_type::rank())
    constexpr reference operator[](Indices... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }

    template <class... Indices>
        requires(std::conjunction_v<std::is_convertible<Indices, index_type>...> &&
                 sizeof...(Indices) == extents_type::rank())
    constexpr const_reference operator[](Indices... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(indices...)];
    }
#endif

    //----------------------------------------------------------------------------------------------
    // Iterators

    constexpr iterator begin() noexcept { return ctr.begin(); }

    constexpr const_iterator begin() const noexcept { return ctr.begin(); }
    constexpr const_iterator cbegin() const noexcept { return ctr.cbegin(); }

    constexpr reverse_iterator rbegin() noexcept { return ctr.rbegin(); }

    constexpr const_reverse_iterator rbegin() const noexcept { return ctr.rbegin(); }

    constexpr const_reverse_iterator crbegin() const noexcept { return ctr.crbegin(); }

    constexpr iterator end() noexcept { return ctr.end(); }

    constexpr const_iterator end() const noexcept { return ctr.end(); }
    constexpr const_iterator cend() const noexcept { return ctr.cend(); }

    constexpr reverse_iterator rend() noexcept { return ctr.rend(); }

    constexpr const_reverse_iterator rend() const noexcept { return ctr.rend(); }

    constexpr const_reverse_iterator crend() const noexcept { return ctr.crend(); }

    //----------------------------------------------------------------------------------------------
    // Access underlying data

    constexpr pointer data() noexcept { return ctr.data(); }
    constexpr const_pointer data() const noexcept { return ctr.data(); }

    constexpr container_type& container() noexcept { return ctr; }
    constexpr const container_type& container() const noexcept { return ctr; }

    //----------------------------------------------------------------------------------------------
    // Return view of data

    constexpr view_type view() noexcept { return view_type(ctr.data(), map.extents()); }

    constexpr const_view_type view() const noexcept
    {
        return const_view_type(ctr.data(), map.extents());
    }

    //----------------------------------------------------------------------------------------------
    // Observers of the multidimensional index space

    constexpr static std::size_t rank() noexcept { return extents_type::rank(); }

    constexpr bool empty() const noexcept { return ctr.empty(); }

    constexpr std::size_t size() const noexcept { return ctr.size(); }

    constexpr difference_type ssize() const noexcept
    {
        return static_cast<difference_type>(ctr.size());
    }

    constexpr std::size_t max_size() const noexcept { return ctr.max_size(); }

    constexpr extents_type extents() const noexcept { return map.extents(); }

    constexpr index_type extent(std::size_t dim) const noexcept
    {
        assert(dim < extents_type::rank());
        return map.extents().extent(dim);
    }

    // Signed extent.
    constexpr difference_type sextent(index_type dim) const noexcept
    {
        assert(dim < extents_type::rank());
        return static_cast<difference_type>(map.extents().extent(dim));
    }

    //----------------------------------------------------------------------------------------------
    // Observers of the mapping

    constexpr mapping_type mapping() const noexcept { return map; }

    constexpr index_type stride(std::size_t dim) const noexcept
    {
        assert(dim < extents_type::rank());
        return map.stride(dim);
    }

    //----------------------------------------------------------------------------------------------
    // Modifiers

    template <class... Exts>
        requires(std::conjunction_v<std::is_convertible<Exts, index_type>...> &&
                 sizeof...(Exts) == extents_type::rank())
    constexpr void resize(Exts... exts) noexcept
    {
        map = mapping_type(extents_type(static_cast<index_type>(exts)...));
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
        for (index_type i = 0; i < static_cast<index_type>(size()); ++i) {
            f(ctr[i]);
        }
        return *this;
    }

    template <class F, class U>
    constexpr MDArray& apply(F f, const U& val) noexcept
    {
        for (index_type i = 0; i < static_cast<index_type>(size()); ++i) {
            f(ctr[i], val);
        }
        return *this;
    }

    template <class F>
    constexpr MDArray& apply(const MDArray& m, F f) noexcept
    {
        assert(view().extents() == m.view().extents());

        for (index_type i = 0; i < static_cast<index_type>(size()); ++i) {
            f(ctr[i], m.ctr[i]);
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
#endif

} // namespace Sci
#endif // SCILIB_MDARRAY_BITS_H
