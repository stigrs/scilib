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
#include <gsl/gsl>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>


namespace Sci {

namespace __Detail {

//--------------------------------------------------------------------------------------------------
// Type traits:

template <class M>
struct Is_mdarray : std::false_type {
};

template <class ElementType, class Extents, class LayoutPolicy, class Container>
struct Is_mdarray<MDArray<ElementType, Extents, LayoutPolicy, Container>> : std::true_type {
};

template <class M>
static constexpr bool Is_mdarray_v = Is_mdarray<M>::value;

template <class ValueType, class Index>
decltype(auto) just_value(Index, ValueType&& t)
{
    return std::forward<ValueType&&>(t);
}

template <class ValueType, std::size_t N>
std::array<ValueType, N> value_to_array(const ValueType& t)
{
    return [&]<std::size_t... Indices>(std::index_sequence<Indices...>)
    {
        return std::array<ValueType, N>{just_value(Indices, t)...};
    }
    (std::make_index_sequence<N>());
}

template <class Container>
struct Container_is_array : std::false_type {
    template <class M>
    static constexpr Container construct(const M& m)
    {
        return Container(m.required_span_size());
    }

    template <class M, class ValueType>
    static constexpr Container construct(const M& m, const ValueType& val)
    {
        return Container(m.required_span_size(), val);
    }
};

template <class ElementType, std::size_t N>
struct Container_is_array<std::array<ElementType, N>> : std::true_type {
    template <class M>
    static constexpr std::array<ElementType, N> construct(const M&)
    {
        return std::array<ElementType, N>();
    }

    template <class M, class ValueType>
    static constexpr std::array<ElementType, N> construct(const M&, const ValueType& val)
    {
        return value_to_array<ElementType, N>(val);
    }
};

template <class Container>
static constexpr bool Container_is_array_v = Container_is_array<Container>::value;

template <class Container>
struct Container_is_vector : std::false_type {
};

template <class ElementType, class Allocator>
struct Container_is_vector<std::vector<ElementType, Allocator>> : std::true_type {
};

template <class Container>
static constexpr bool Container_is_vector_v = Container_is_vector<Container>::value;

//--------------------------------------------------------------------------------------------------
// Bounds checking:

template <class IndexType, class From>
    requires(std::is_integral_v<From>)
constexpr bool __is_index_in_extent(IndexType extent, From value)
{
    using index_type = std::common_type_t<IndexType, From>;
    return value >= 0 && gsl::narrow_cast<index_type>(value) < gsl::narrow_cast<index_type>(extent);
}

template <std::size_t... Idxs, class Extents, class... From>
constexpr bool
__check_bounds_impl(std::index_sequence<Idxs...>, const Extents& exts, From... values)
{
    return (__is_index_in_extent(exts.extent(Idxs), values) && ...);
}

template <class Extents, class... From>
constexpr bool __check_bounds(const Extents& exts, From... values)
{
    return __check_bounds_impl(std::make_index_sequence<Extents::rank()>(), exts, values...);
}

} // namespace __Detail

// Dense multidimensional array class for numerical computing using mdspan
// for views.
//
// Storage order can be either row-major (layout_right; default) or
// column-major (layout_left).
//
// The design is based on the proposed mdarray (P1684R5) that is specified
// in the following paper:
//   https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p1684r5.html
//
template <class ElementType, class Extents, class LayoutPolicy, class Container>
    requires __Detail::Is_extents_v<Extents>
class MDArray {
public:
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using container_type = Container;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using element_type = ElementType;
    using mdspan_type = Kokkos::mdspan<element_type, extents_type, layout_type>;
    using const_mdspan_type = Kokkos::mdspan<const element_type, extents_type, layout_type>;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using pointer = decltype(std::to_address(std::declval<container_type>().begin()));
    using const_pointer = decltype(std::to_address(std::declval<container_type>().cbegin()));
    using reference = typename container_type::reference;
    using const_reference = typename container_type::const_reference;

    static constexpr rank_type rank() noexcept { return extents_type::rank(); }
    static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
    static constexpr size_type static_extent(rank_type r) noexcept
    {
        Expects(r < extents_type::rank());
        return extents_type::static_extent(r);
    }

    constexpr index_type extent(rank_type r) const
    {
        Expects(r < extents_type::rank());
        return map.extents().extent(r);
    }

    // [MDArray.ctors], MDArray constructors

    constexpr MDArray() requires(extents_type::rank_dynamic() != 0) = default;
    constexpr MDArray(const MDArray&) = default;
    constexpr MDArray(MDArray&&) = default;

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 std::is_constructible_v<extents_type, OtherIndexTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>) )
    constexpr explicit MDArray(OtherIndexTypes... exts)
        : map(extents_type(static_cast<index_type>(std::move(exts))...)),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts) requires(
        std::is_constructible_v<mapping_type, const extents_type&> &&
        (std::is_constructible_v<container_type, std::size_t> ||
         __Detail::Container_is_array_v<container_type>) )
        : map(exts), ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m) requires(
        std::is_constructible_v<container_type, std::size_t> ||
        __Detail::Container_is_array_v<container_type>)
        : map(m), ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, const value_type& val) requires(
        std::is_constructible_v<mapping_type, const extents_type&> &&
        (std::is_constructible_v<container_type, std::size_t> ||
         __Detail::Container_is_array_v<container_type>) )
        : map(exts), ctr(__Detail::Container_is_array<container_type>::construct(map, val))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m, const value_type& val) requires(
        std::is_constructible_v<container_type, std::size_t> ||
        __Detail::Container_is_array_v<container_type>)
        : map(m), ctr(__Detail::Container_is_array<container_type>::construct(map, val))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, const container_type& c) requires(
        std::is_constructible_v<mapping_type, const extents_type&>)
        : map(exts), ctr(c)
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m, const container_type& c) : map(m), ctr(c)
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, container_type&& c) requires(
        std::is_constructible_v<mapping_type, const extents_type&>)
        : map(exts), ctr(std::move(c))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m, container_type&& c) : map(m), ctr(std::move(c))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    template <class U>
    constexpr MDArray(std::initializer_list<U>) = delete;

    template <class U>
    constexpr MDArray& operator=(std::initializer_list<U>) = delete;

    constexpr MDArray(
        __Detail::MDArray_initializer<element_type, extents_type::rank()>
            init) requires((!std::is_same_v<layout_type, Kokkos::layout_stride>) &&std::
                               is_same_v<container_type, std::vector<element_type>>)
        : map(extents_type(__Detail::derive_extents<extents_type::rank()>(init)))
    {
        ctr.reserve(map.required_span_size());
        __Detail::insert_flat(init, ctr);

        if constexpr (std::is_same_v<layout_type, Kokkos::layout_left>) { // need to transpose data
            MDArray<element_type, extents_type, Kokkos::layout_right, container_type> tmp(extents(),
                                                                                         ctr);
            (*this) = tmp.to_mdspan();
        }
    }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class OtherContainer>
        requires(std::is_constructible_v<Container, const OtherContainer&>&&
                     std::is_constructible_v<extents_type, OtherExtents>)
    constexpr MDArray(
        const MDArray<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainer>& other)
        : map(extents_type(other.extents())),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(Kokkos::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices)
        {
#if MDSPAN_USE_BRACKET_OPERATOR
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other[static_cast<index_type>(std::move(indices))...];
#else
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other(static_cast<index_type>(std::move(indices))...);
#endif
        };
        for_each_in_extents(copy_fn, other);
    }

    template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class Accessor>
        requires(std::is_constructible_v<extents_type, OtherExtents>&&
                     std::is_constructible_v<value_type, typename Accessor::reference> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>) )
    constexpr MDArray(
        const Kokkos::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor>& other)
        : map(extents_type(other.extents())),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(Kokkos::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
#if _MSC_VER
#pragma warning(disable : 4834)
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices)
        {
#if MDSPAN_USE_BRACKET_OPERATOR
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other[static_cast<index_type>(std::move(indices))...];
#else
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other(static_cast<index_type>(std::move(indices))...);
#endif
        };
#pragma warning(default : 4834)
#endif // _MSC_VER
        for_each_in_extents(copy_fn, other);
    }

    ~MDArray() = default;

    // [MDArray.ctors.alloc], MDArray constructors with allocators

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&>&&
                     std::is_constructible_v<container_type, std::size_t, Alloc>)
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

    template <class Alloc>
    constexpr MDArray(const extents_type& exts, const value_type& val, const Alloc& a) requires(
        std::is_constructible_v<mapping_type, const extents_type&>&&
            std::is_constructible_v<container_type, std::size_t, value_type, Alloc>)
        : map(exts), ctr(map.required_span_size(), val, a)
    {
    }

    template <class Alloc>
    constexpr MDArray(const mapping_type& m, const value_type& val, const Alloc& a) requires(
        std::is_constructible_v<container_type, std::size_t, value_type, Alloc>)
        : map(m), ctr(map.required_span_size(), val, a)
    {
    }

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&>&&
                     std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const extents_type& exts, const container_type& c, const Alloc& a)
        : map(exts), ctr(c, a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const mapping_type& m, container_type& c, const Alloc& a) : map(m), ctr(c, a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&>&&
                     std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const extents_type& exts, container_type&& c, Alloc& a)
        : map(exts), ctr(std::move(c), a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const mapping_type& m, container_type&& c, const Alloc& a)
        : map(m), ctr(std::move(c), a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class OtherContainer,
              class Alloc>
        requires(std::is_constructible_v<Container, OtherContainer, Alloc>&&
                     std::is_constructible_v<extents_type, OtherExtents>)
    constexpr MDArray(
        const MDArray<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainer>& other,
        const Alloc& a)
        : map(extents_type(other.extents())), ctr(map.required_span_size(), a)
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(Kokkos::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices)
        {
#if MDSPAN_USE_BRACKET_OPERATOR
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other[static_cast<index_type>(std::move(indices))...];
#else
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other(static_cast<index_type>(std::move(indices))...);
#endif
        };
        for_each_in_extents(copy_fn, other);
    }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class Accessor,
              class Alloc>
        requires(std::is_constructible_v<extents_type, OtherExtents>&&
                     std::is_constructible_v<value_type, typename Accessor::reference>&&
                         std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(
        const Kokkos::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor>& other,
        const Alloc& a)
        : map(extents_type(other.extents())), ctr(map.required_span_size(), a)
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(Kokkos::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices)
        {
#if MDSPAN_USE_BRACKET_OPERATOR
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other[static_cast<index_type>(std::move(indices))...];
#else
            ctr[map(static_cast<index_type>(std::move(indices))...)] =
                other(static_cast<index_type>(std::move(indices))...);
#endif
        };
        for_each_in_extents(copy_fn, other);
    }

    constexpr MDArray& operator=(const MDArray&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    // [MDArray.members], MDArray members

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference operator()(OtherIndexTypes... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    MDSPAN_FORCE_INLINE_FUNCTION const_reference
    operator()(OtherIndexTypes... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>&&
                         extents_type::rank() == 1)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference operator[](OtherIndexType indx) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indx));
        return ctr[map(static_cast<index_type>(std::move(indx)))];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>&&
                         extents_type::rank() == 1)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](OtherIndexType indx) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indx));
        return ctr[map(static_cast<index_type>(std::move(indx)))];
    }

#if MDSPAN_USE_BRACKET_OPERATOR
    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference operator[](OtherIndexTypes... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](OtherIndexTypes... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }
#endif

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference
    operator[](const std::array<OtherIndexType, rank()>& indices) noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>)
        {
            return map(indices[Indxs]...);
        }
        (std::make_index_sequence<rank()>());
        return ctr[map_fn];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](const std::array<OtherIndexType, rank()>& indices) const noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>)
        {
            return map(indices[Indxs]...);
        }
        (std::make_index_sequence<rank()>());
        return ctr[map_fn];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference
    operator[](std::span<OtherIndexType, rank()> indices) noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>)
        {
            return map(indices[Indxs]...);
        }
        (std::make_index_sequence<rank()>());
        return ctr[map_fn];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type>&&
                     std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](std::span<OtherIndexType, rank()> indices) const noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>)
        {
            return map(indices[Indxs]...);
        }
        (std::make_index_sequence<rank()>());
        return ctr[map_fn];
    }

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    constexpr reference at(OtherIndexTypes... indices) noexcept
    {
        Expects(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    constexpr const_reference at(OtherIndexTypes... indices) const noexcept
    {
        Expects(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    constexpr const extents_type& extents() const noexcept { return map.extents(); }
    constexpr const mapping_type& mapping() const noexcept { return map; }

    constexpr bool empty() const noexcept
    {
        return [&]<std::size_t... IndexTypes>(std::index_sequence<IndexTypes...>)
        {
            return (rank() > 0) &&
                   ((map.extents().extent(IndexTypes) == index_type{0}) || ... || false);
        }
        (std::make_index_sequence<rank()>());
    }

    constexpr size_type size() const noexcept
    {
        return [&]<std::size_t... IndexTypes>(std::index_sequence<IndexTypes...>)
        {
            return ((static_cast<size_type>(map.extents().extent(IndexTypes))) * ... *
                    size_type{1});
        }
        (std::make_index_sequence<rank()>());
    }

    constexpr size_type container_size() const noexcept { return ctr.size(); }

    constexpr pointer container_data() noexcept { return ctr.data(); }
    constexpr const_pointer container_data() const noexcept { return ctr.data(); }

    container_type&& extract_container() noexcept { return std::move(ctr); }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutType,
              class OtherAccessorType>
        requires(std::is_assignable_v<
                 mdspan_type,
                 Kokkos::mdspan<OtherElementType, OtherExtents, OtherLayoutType, OtherAccessorType>>)
    constexpr
    operator Kokkos::mdspan<OtherElementType, OtherElementType, OtherLayoutType, OtherAccessorType>()
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return mdspan_type(container_data(), map);
    }

    template <class OtherAccessorType = Kokkos::default_accessor<element_type>>
        requires(std::is_assignable_v<
                 mdspan_type,
                 Kokkos::mdspan<element_type, extents_type, layout_type, OtherAccessorType>>)
    constexpr Kokkos::mdspan<element_type, extents_type, layout_type, OtherAccessorType>
    to_mdspan(const OtherAccessorType& a = Kokkos::default_accessor<element_type>())
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return Kokkos::mdspan<element_type, extents_type, layout_type, OtherAccessorType>(
            container_data(), map, a);
    }

    template <class OtherAccessorType = Kokkos::default_accessor<const element_type>>
        requires(std::is_assignable_v<
                 const_mdspan_type,
                 Kokkos::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>>)
    constexpr Kokkos::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>
    to_mdspan(const OtherAccessorType& a = Kokkos::default_accessor<const element_type>()) const
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return Kokkos::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>(
            container_data(), map, a);
    }

    static constexpr bool is_always_unique() noexcept { return mapping_type::is_always_unique(); };

    static constexpr bool is_always_exhaustive() noexcept
    {
        return mapping_type::is_always_exhaustive();
    };

    static constexpr bool is_always_strided() noexcept
    {
        return mapping_type::is_always_strided();
    };

    constexpr bool is_unique() const noexcept { return map.is_unique(); };
    constexpr bool is_exhaustive() const noexcept { return map.is_exhaustive(); };
    constexpr bool is_strided() const noexcept { return map.is_strided(); };

    constexpr index_type stride(rank_type r) const
    {
        Expects(r < extents_type::rank());
        return map.stride(r);
    };

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 std::is_constructible_v<extents_type, OtherIndexTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type> &&
                 __Detail::Container_is_vector_v<container_type> )
    constexpr void resize(OtherIndexTypes... exts) noexcept
    {
        map = mapping_type(extents_type(std::move(exts)...));
        ctr = container_type(map.required_span_size());
    }

    friend constexpr void swap(MDArray& x, MDArray& y) noexcept
    {
        std::swap(x.ctr, y.ctr);
        std::swap(x.map, y.map);
    }

    template <class Callable>
    constexpr MDArray& apply(Callable&& f) noexcept
    {
        auto apply_fn = [&]<class... IndexTypes>(IndexTypes... indices)
        {
            std::forward<Callable>(f)(ctr[map(static_cast<index_type>(std::move(indices))...)]);
        };
        for_each_in_extents(apply_fn, extents(), layout_type{});

        return *this;
    }

    template <class Callable, class ValueType>
    constexpr MDArray& apply(Callable&& f, const ValueType& val) noexcept
    {
        auto apply_fn = [&]<class... IndexTypes>(IndexTypes... indices)
        {
            std::forward<Callable>(f)(ctr[map(static_cast<index_type>(std::move(indices))...)],
                                      val);
        };
        for_each_in_extents(apply_fn, extents(), layout_type{});

        return *this;
    }

    template <class Callable>
    constexpr MDArray& apply(const MDArray& m, Callable&& f) noexcept
    {
        Expects(extents() == m.extents());

        auto apply_fn = [&]<class... IndexTypes>(IndexTypes... indices)
        {
#if _MSC_VER
#pragma warning(disable : 4834)
#if MDSPAN_USE_BRACKET_OPERATOR
            std::forward<Callable>(f)(ctr[map(static_cast<index_type>(std::move(indices))...)],
                                      m[static_cast<index_type>(std::move(indices))...]);
#else
            std::forward<Callable>(f)(ctr[map(static_cast<index_type>(std::move(indices))...)],
                                      m(static_cast<index_type>(std::move(indices))...));
#endif
#pragma warning(default : 4834)
#endif // _MSC_VER
        };
        for_each_in_extents(apply_fn, extents(), layout_type{});

        return *this;
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a = value; });
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator+=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a += value; });
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator-=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a -= value; });
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator*=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a *= value; });
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator/=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a /= value; });
    }

    template <class OtherElementType>
        requires(std::is_convertible_v<element_type, OtherElementType>)
    constexpr MDArray& operator%=(const OtherElementType& value) noexcept
    {
        return apply([&](element_type& a) { a %= value; });
    }

    constexpr MDArray& operator+=(const MDArray& m) noexcept
    {
        return apply(m, [](element_type& a, const element_type& b) { a += b; });
    }

    constexpr MDArray& operator-=(const MDArray& m) noexcept
    {
        return apply(m, [](element_type& a, const element_type& b) { a -= b; });
    }

private:
    mapping_type map;
    container_type ctr;
};

template <class IndexType, std::size_t... Extents, class Container>
MDArray(const Kokkos::extents<IndexType, Extents...>&, const Container&)
    -> MDArray<typename Container::value_type,
               Kokkos::extents<IndexType, Extents...>,
               Kokkos::layout_right,
               Container>;

template <class Mapping, class Container>
MDArray(const Mapping&, const Container&) -> MDArray<typename Container::value_type,
                                                     typename Mapping::extents_type,
                                                     typename Mapping::layout_type,
                                                     Container>;

template <class IndexType, std::size_t... Extents, class Container>
MDArray(const Kokkos::extents<IndexType, Extents...>&, Container&&)
    -> MDArray<typename Container::value_type,
               Kokkos::extents<IndexType, Extents...>,
               Kokkos::layout_right,
               Container>;

template <class Mapping, class Container>
MDArray(const Mapping&, Container&&) -> MDArray<typename Container::value_type,
                                                typename Mapping::extents_type,
                                                typename Mapping::layout_type,
                                                Container>;

template <class ElementType, class Extents, class Layout, class Accessor>
MDArray(const Kokkos::mdspan<ElementType, Extents, Layout, Accessor>&)
    -> MDArray<std::remove_cv_t<ElementType>, Extents, Layout>;

template <class IndexType, std::size_t... Extents, class Container, class Alloc>
MDArray(const Kokkos::extents<IndexType, Extents...>&, const Container&, const Alloc&)
    -> MDArray<typename Container::value_type,
               Kokkos::extents<IndexType, Extents...>,
               Kokkos::layout_right,
               Container>;

template <class Mapping, class Container, class Alloc>
MDArray(const Mapping&, const Container&, const Alloc&) -> MDArray<typename Container::value_type,
                                                                   typename Mapping::extents_type,
                                                                   typename Mapping::layout_type,
                                                                   Container>;

template <class IndexType, std::size_t... Extents, class Container, class Alloc>
MDArray(const Kokkos::extents<IndexType, Extents...>&, Container&&, const Alloc&)
    -> MDArray<typename Container::value_type,
               Kokkos::extents<IndexType, Extents...>,
               Kokkos::layout_right,
               Container>;

template <class Mapping, class Container, class Alloc>
MDArray(const Mapping&, Container&&, const Alloc&) -> MDArray<typename Container::value_type,
                                                              typename Mapping::extents_type,
                                                              typename Mapping::layout_type,
                                                              Container>;

template <class ElementType, class Extents, class Layout, class Accessor, class Alloc>
MDArray(const Kokkos::mdspan<ElementType, Extents, Layout, Accessor>&, const Alloc&)
    -> MDArray<std::remove_cv_t<ElementType>, Extents, Layout>;

} // namespace Sci

#endif // SCILIB_MDArray_BITS_H
