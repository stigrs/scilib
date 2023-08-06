// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#include "support.h"
#include <array>
#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Sci {
namespace stdex = std::experimental;

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
decltype(auto) just_value(Index, ValueType&& t) { return std::forward<ValueType&&>(t); }

template <class ValueType, std::size_t N>
std::array<ValueType, N> value_to_array(const ValueType& t) 
{
    return [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
        return std::array<ValueType, N>{just_value(Indices, t)...};
    }(std::make_index_sequence<N>());
}

template <class Container>
struct Container_is_array : std::false_type {
    template <class M>
    static constexpr Container construct(const M& m) { return Container(m.required_span_size()); }

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

//--------------------------------------------------------------------------------------------------
// Bounds checking:

template <class Extents, class... Dims>
    requires __Detail::Is_extents_v<Extents>
inline bool __check_bounds(const Extents& exts, Dims... dims)
{
    using index_type = typename Extents::index_type;

    std::array<index_type, Extents::rank()> indexes{static_cast<index_type>(dims)...};
    bool result = true;
    for (std::size_t i = 0; i < indexes.size(); ++i) {
        if (!(indexes[i] >= 0 && indexes[i] < exts.extent(i))) {
            result = false;
        }
    }
    return result;
}

//--------------------------------------------------------------------------------------------------
// Size of extents:

template <class Extents>
    requires __Detail::Is_extents_v<Extents>
inline std::size_t size_of_extents(const Extents& exts) 
{
    std::size_t size = 1;
    for (std::size_t r = 0; r < exts.rank(); ++r) {
        size *= exts.extent(r);
    }
    return size;
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
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_type>;
    using const_mdspan_type = stdex::mdspan<const element_type, extents_type, layout_type>;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using pointer = decltype(std::to_address(std::declval<container_type>().begin()));
    using const_pointer = decltype(std::to_address(std::declval<container_type>().cbegin()));
    using reference = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

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
                  __Detail::Container_is_array_v<container_type>))
    constexpr explicit MDArray(OtherIndexTypes... exts)
        : map(extents_type(static_cast<index_type>(std::move(exts))...)),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    template <class... OtherIndexTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 std::is_constructible_v<extents_type, OtherIndexTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>))
    constexpr explicit MDArray(const container_type& c, OtherIndexTypes... exts)
        : map(extents_type(static_cast<index_type>(std::move(exts))...)), ctr(c)
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts)
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>) )
        : map(exts), ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m)
        requires(std::is_constructible_v<container_type, std::size_t> ||
                 __Detail::Container_is_array_v<container_type>)
        : map(m), ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, const value_type& val)
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>) )
        : map(exts), ctr(__Detail::Container_is_array<container_type>::construct(map, val))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m, const value_type& val)
        requires(std::is_constructible_v<container_type, std::size_t> ||
                 __Detail::Container_is_array_v<container_type>)
        : map(m), ctr(__Detail::Container_is_array<container_type>::construct(map, val))
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, const container_type& c)
        requires(std::is_constructible_v<mapping_type, const extents_type&>)
        : map(exts), ctr(c)
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const mapping_type& m, const container_type& c) : map(m), ctr(c)
    {
        Expects(gsl::narrow_cast<size_type>(map.required_span_size()) <= ctr.size());
    }

    constexpr MDArray(const extents_type& exts, container_type&& c) 
        requires(std::is_constructible_v<mapping_type, const extents_type&>)
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

    constexpr MDArray(__Detail::MDArray_initializer<element_type, extents_type::rank()> init) 
        requires((!std::is_same_v<layout_type, stdex::layout_stride>) && 
                 std::is_same_v<container_type, std::vector<element_type>>)
        : map(extents_type(__Detail::derive_extents<extents_type::rank()>(init)))
    {
        ctr.reserve(map.required_span_size());
        __Detail::insert_flat(init, ctr);

        if constexpr (std::is_same_v<layout_type, stdex::layout_left>) { // need to transpose data
            MDArray<element_type, extents_type, stdex::layout_right, container_type> tmp(extents(),
                                                                                         ctr);
            (*this) = tmp.to_mdspan();
        }
    }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutPolicy,
              class OtherContainer>
        requires(std::is_constructible_v<Container, const OtherContainer&> &&
                 std::is_constructible_v<extents_type, OtherExtents>) 
    constexpr MDArray(
        const MDArray<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainer>& other)
        : map(extents_type(__Detail::extents(other))),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices) {
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
        requires(std::is_constructible_v<extents_type, OtherExtents> &&
                 std::is_constructible_v<value_type, typename Accessor::reference> &&
                 (std::is_constructible_v<container_type, std::size_t> ||
                  __Detail::Container_is_array_v<container_type>) )
    constexpr MDArray(
        const stdex::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor>& other)
        : map(extents_type(__Detail::extents(other))),
          ctr(__Detail::Container_is_array<container_type>::construct(map))
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices) {
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

    ~MDArray() = default;

    // [MDArray.ctors.alloc], MDArray constructors with allocators

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
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
    constexpr MDArray(const extents_type& exts, const value_type& val, const Alloc& a)
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
                 std::is_constructible_v<container_type, std::size_t, value_type, Alloc>)
        : map(exts), ctr(map.required_span_size(), val, a)
    {
    }

    template <class Alloc>
    constexpr MDArray(const mapping_type& m, const value_type& val, const Alloc& a)
        requires(std::is_constructible_v<container_type, std::size_t, value_type, Alloc>)
        : map(m), ctr(map.required_span_size(), val, a)
    {
    }

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const extents_type& exts, const container_type& c, const Alloc& a)
        : map(exts), ctr(c, a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const mapping_type& m, container_type& c, const Alloc& a)
        : map(m), ctr(c, a)
    {
        Expects(ctr.size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, const extents_type&> &&
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
        requires(std::is_constructible_v<Container, OtherContainer, Alloc> &&
                 std::is_constructible_v<extents_type, OtherExtents>)
    constexpr MDArray(
        const MDArray<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainer>& other,
        const Alloc& a)
        : map(extents_type(__Detail::extents(other))), ctr(map.required_span_size(), a)
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices) {
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
        requires(std::is_constructible_v<extents_type, OtherExtents> &&
                 std::is_constructible_v<value_type, typename Accessor::reference> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(
        const stdex::mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, Accessor>& other,
        const Alloc& a)
        : map(extents_type(__Detail::extents(other))),
          ctr(map.required_span_size(), a)
    {
        for (rank_type r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<size_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<size_type>(other.extent(r)));
        }
        auto copy_fn = [&]<class... OtherIndexTypes>(OtherIndexTypes... indices) {
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

#if MDSPAN_USE_PAREN_OPERATOR
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
#endif

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType> &&
                 extents_type::rank() == 1)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference operator[](OtherIndexType indx) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indx));
        return ctr[map(static_cast<index_type>(std::move(indx)))];
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType> &&
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

    template <class... SizeTypes>
        requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
                 (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
                 sizeof...(OtherIndexTypes) == extents_type::rank())
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](SizeTypes... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }
#endif

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference
    operator[](const std::array<OtherIndexType, rank()>& indices) noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>) {
            return map(indices[Indxs]...);
        }(std::make_index_sequence<rank()>());
        return ctr[map_fn]; 
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](const std::array<OtherIndexType, rank()>& indices) const noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>) {
            return map(indices[Indxs]...);
        }(std::make_index_sequence<rank()>());
        return ctr[map_fn]; 
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr reference
    operator[](std::span<OtherIndexType, rank()> indices) noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>) {
            return map(indices[Indxs]...);
        }(std::make_index_sequence<rank()>());
        return ctr[map_fn]; 
    }

    template <class OtherIndexType>
        requires(std::is_convertible_v<OtherIndexType, index_type> &&
                 std::is_nothrow_constructible_v<index_type, OtherIndexType>)
    MDSPAN_FORCE_INLINE_FUNCTION constexpr const_reference
    operator[](std::span<OtherIndexType, rank()> indices) const noexcept
    {
        auto map_fn = [&]<std::size_t... Indxs>(std::index_sequence<Indxs...>) {
            return map(indices[Indxs]...);
        }(std::make_index_sequence<rank()>());
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
        return [&]<std::size_t... IndexTypes>(std::index_sequence<IndexTypes...>) {
            return (rank() > 0) &&
                   ((map.extents().extent(IndexTypes) == index_type{0}) || ... || false);
        }(std::make_index_sequence<rank()>());
    }

    constexpr size_type size() const noexcept 
    {
        return [&]<std::size_t... IndexTypes>(std::index_sequence<IndexTypes...>) {
            return ((static_cast<size_type>(map.extents().extent(IndexTypes))) * ... *
                    size_type{1});
        }(std::make_index_sequence<rank()>());
    }

    constexpr size_type container_size() const noexcept { return ctr.size(); }

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

    constexpr pointer container_data() noexcept { return ctr.data(); }
    constexpr const_pointer container_data() const noexcept { return ctr.data(); }

    container_type&& extract_container() noexcept { return std::move(ctr); }

    template <class OtherElementType,
              class OtherExtents,
              class OtherLayoutType,
              class OtherAccessorType>
        requires(std::is_assignable_v<
                 mdspan_type,
                 stdex::mdspan<OtherElementType, OtherExtents, OtherLayoutType, OtherAccessorType>>)
    constexpr
    operator stdex::mdspan<OtherElementType, OtherElementType, OtherLayoutType, OtherAccessorType>()
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return mdspan_type(container_data(), map);
    }

    template <class OtherAccessorType = stdex::default_accessor<element_type>>
        requires(std::is_assignable_v<
                 mdspan_type,
                 stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>>)
    constexpr stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>
    to_mdspan(const OtherAccessorType& a = stdex::default_accessor<element_type>())
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>(
            container_data(), map, a);
    }

    template <class OtherAccessorType = stdex::default_accessor<const element_type>>
        requires(std::is_assignable_v<
                 const_mdspan_type,
                 stdex::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>>)
    constexpr stdex::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>
    to_mdspan(const OtherAccessorType& a = stdex::default_accessor<const element_type>()) const
    {
        Expects(container_size() >= gsl::narrow_cast<size_type>(map.required_span_size()));
        return stdex::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>(
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
                 (!__Detail::Container_is_array_v<container_type>))
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
        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            std::forward<Callable>(f)(ctr[i]);
        }
        return *this;
    }

    template <class Callable, class ValueType>
    constexpr MDArray& apply(Callable&& f, const ValueType& val) noexcept
    {
        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            std::forward<Callable>(f)(ctr[i], val);
        }
        return *this;
    }

    template <class Callable>
    constexpr MDArray& apply(const MDArray& m, Callable&& f) noexcept
    {
        Expects(extents() == m.extents());

        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            std::forward<Callable>(f)(ctr[i], m.ctr[i]);
        }
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

} // namespace Sci

#endif // SCILIB_MDARRAY_BITS_H
