// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_BITS_H
#define SCILIB_MDARRAY_BITS_H

#include "support.h"
#include <array>
#include <cassert>
#include <gsl/gsl>
#include <initializer_list>
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

template <class Container>
struct Container_is_array : std::false_type {
    static constexpr Container construct(std::size_t size) { return Container(size); }
};

template <class ElementType, std::size_t N>
struct Container_is_array<std::array<ElementType, N>> : std::true_type {
    static constexpr std::array<ElementType, N> construct(std::size_t)
    {
        return std::array<ElementType, N>();
    }
};

//--------------------------------------------------------------------------------------------------
// Bounds checking:

template <class Extents, class... Dims>
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
    requires __Detail::Is_extents_v<Extents>
class MDArray {
public:
    using element_type = ElementType;
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using container_type = Container;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using view_type = stdex::mdspan<element_type, extents_type, layout_type>;
    using const_view_type = stdex::mdspan<const element_type, extents_type, layout_type>;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename extents_type::index_type;
    using rank_type = typename extents_type::rank_type;
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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(c.size()));
    }

    // clang-format off
    constexpr MDArray(const container_type& c, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type>)
        : map(exts), ctr(c)
    // clang-format on
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(c.size()));
    }

    constexpr MDArray(const container_type& c, const mapping_type& m) : map(m), ctr(c)
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(c.size()));
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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(ctr.size()));
    }

    // clang-format off
    constexpr MDArray(container_type&& c, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type>)
        : map(exts), ctr(std::move(c))
    // clang-format on
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(ctr.size()));
    }

    constexpr MDArray(container_type&& c, const mapping_type& m) : map(m), ctr(std::move(c))
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(ctr.size()));
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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(init.size()));
    }

    // clang-format off
    constexpr MDArray(std::initializer_list<element_type> init, const extents_type& exts) 
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::initializer_list<element_type>>)
        : map(exts), ctr(init)
    // clang-format on
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(init.size()));
    }

    // clang-format off
    constexpr MDArray(std::initializer_list<element_type> init, const mapping_type& m) 
        requires(std::is_constructible_v<container_type, std::initializer_list<element_type>>)
        : map(m), ctr(init)
    // clang-format on
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(init.size()));
    }

    template <class U>
    constexpr MDArray(std::initializer_list<U>) = delete;

    template <class U>
    constexpr MDArray& operator=(std::initializer_list<U>) = delete;

    // clang-format off
    constexpr MDArray(__Detail::MDArray_initializer<element_type, extents_type::rank()> init) 
        requires((!std::is_same_v<layout_type, stdex::layout_stride>) && 
                 std::is_same_v<container_type, std::vector<element_type>>)
        : map(extents_type(__Detail::derive_extents<extents_type::rank()>(init)))
    // clang-format on
    {
        ctr.reserve(map.required_span_size());
        __Detail::insert_flat(init, ctr);

        if constexpr (std::is_same_v<layout_type, stdex::layout_left>) { // need to transpose data
            MDArray<element_type, extents_type, stdex::layout_right, container_type> tmp(ctr,
                                                                                         extents());
            (*this) = tmp.view();
        }
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
        : map(extents_type(__Detail::extents(other))),
          ctr(__Detail::Container_is_array<container_type>::construct(map.required_span_size()))
    {
        static_assert(other.rank() <= 7);
        Expects(ctr.size() == other.size());
        for (std::size_t r = 0; r < other.rank(); ++r) {
            Expects(static_extent(r) == gsl::narrow_cast<index_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<index_type>(other.extent(r)));
        }
        copy(other, view());
    }

    ~MDArray() = default;

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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(c.size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(const container_type& c, const mapping_type& m, const Alloc& a)
        : map(m), ctr(c, a)
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(c.size()));
    }

    // clang-format off
    template <class Alloc>
        requires(std::is_constructible_v<mapping_type, extents_type> &&
                 std::is_constructible_v<container_type, std::size_t, Alloc>)
    // clang-format on
    constexpr MDArray(container_type&& c, const extents_type& exts, const Alloc& a)
        : map(exts), ctr(c, a)
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(ctr.size()));
    }

    template <class Alloc>
        requires(std::is_constructible_v<container_type, std::size_t, Alloc>)
    constexpr MDArray(container_type&& c, const mapping_type& m, const Alloc& a) : map(m), ctr(c, a)
    {
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(ctr.size()));
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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(init.size()));
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
        Expects(map.required_span_size() == gsl::narrow_cast<index_type>(init.size()));
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
            Expects(static_extent(r) == gsl::narrow_cast<index_type>(stdex::dynamic_extent) ||
                    static_extent(r) == gsl::narrow_cast<index_type>(other.extent(r)));
        }
        copy(other, view());
    }

    constexpr MDArray& operator=(const MDArray&) = default;
    constexpr MDArray& operator=(MDArray&&) = default;

    // [MDArray.members], MDArray members

#if MDSPAN_USE_PAREN_OPERATOR
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr reference operator()(SizeTypes... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr const_reference operator()(SizeTypes... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }
#endif

    template <class SizeType>
        requires(std::is_convertible_v<SizeType, index_type>&& extents_type::rank() == 1)
    constexpr reference operator[](SizeType indx) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indx));
        return ctr[map(static_cast<index_type>(std::move(indx)))];
    }

    template <class SizeType>
        requires(std::is_convertible_v<SizeType, index_type>&& extents_type::rank() == 1)
    constexpr const_reference operator[](SizeType indx) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indx));
        return ctr[map(static_cast<index_type>(std::move(indx)))];
    }

#if MDSPAN_USE_BRACKET_OPERATOR
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr reference operator[](SizeTypes... indices) noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr const_reference operator[](SizeTypes... indices) const noexcept
    {
        assert(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }
#endif

    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr reference at(SizeTypes... indices) noexcept
    {
        Expects(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 sizeof...(SizeTypes) == extents_type::rank())
    constexpr const_reference at(SizeTypes... indices) const noexcept
    {
        Expects(__Detail::__check_bounds(map.extents(), indices...));
        return ctr[map(static_cast<index_type>(std::move(indices))...)];
    }

    static constexpr rank_type rank() noexcept { return extents_type::rank(); }
    static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
    static constexpr index_type static_extent(std::size_t r) noexcept
    {
        return extents_type::static_extent(r);
    }

    constexpr const extents_type& extents() const noexcept { return map.extents(); }
    constexpr index_type extent(std::size_t r) const
    {
        Expects(r < extents_type::rank());
        return map.extents().extent(r);
    }

    constexpr bool empty() const noexcept { return ctr.empty(); }
    constexpr std::size_t size() const noexcept { return ctr.size(); }

    // [MDArray.members.iterators], iterators over the data

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

    // [MDArray.members.views], views of the data

    constexpr pointer data() noexcept { return ctr.data(); }
    constexpr const_pointer data() const noexcept { return ctr.data(); }

    template <class OtherAccessorType = stdex::default_accessor<element_type>>
        requires(std::is_same_v<element_type, typename OtherAccessorType::element_type>)
    constexpr stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>
    view(const OtherAccessorType& a = stdex::default_accessor<element_type>())
    {
        return stdex::mdspan<element_type, extents_type, layout_type, OtherAccessorType>(data(),
                                                                                         map, a);
    }

    template <class OtherAccessorType = stdex::default_accessor<const element_type>>
        requires(std::is_same_v<const element_type, typename OtherAccessorType::element_type>)
    constexpr stdex::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>
    view(const OtherAccessorType& a = stdex::default_accessor<const element_type>()) const
    {
        return stdex::mdspan<const element_type, extents_type, layout_type, OtherAccessorType>(
            data(), map, a);
    }

    // [MDArray.members.obs], observers of the mapping

    constexpr const mapping_type& mapping() const noexcept { return map; }

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

    constexpr index_type stride(std::size_t r) const
    {
        Expects(r < extents_type::rank());
        return map.stride(r);
    };

    // [MDArray.members.modifiers], MDArray modifiers

    // clang-format off
    template <class... SizeTypes>
        requires(std::conjunction_v<std::is_convertible<SizeTypes, index_type>...> &&
                 std::is_constructible_v<extents_type, SizeTypes...> &&
                 std::is_constructible_v<mapping_type, extents_type> &&
                 (!__Detail::Container_is_array<container_type>::value))
    // clang-format on
    constexpr void resize(SizeTypes... exts) noexcept
    {
        map = mapping_type(extents_type(exts...));
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
        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            f(ctr[i]);
        }
        return *this;
    }

    template <class F, class U>
    constexpr MDArray& apply(F f, const U& val) noexcept
    {
        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            f(ctr[i], val);
        }
        return *this;
    }

    template <class F>
    constexpr MDArray& apply(const MDArray& m, F f) noexcept
    {
        Expects(extents() == m.extents());

        for (index_type i = 0; i < gsl::narrow_cast<index_type>(size()); ++i) {
            f(ctr[i], m.ctr[i]);
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
