// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SUPPORT_H
#define SCILIB_MDARRAY_SUPPORT_H

#include <array>
#include <type_traits>

#if _MSC_VER >= 1927
#include <concepts>
#define STD_CONVERTIBLE_TO(X) std::convertible_to<X>
#else
#define STD_CONVERTIBLE_TO(X) Sci::__Detail::convertible_to<X>
#endif

namespace Sci {
namespace __Detail {

template <class From, class To>
concept convertible_to = std::is_convertible_v<From, To> && requires
{
    static_cast<To>(std::declval<From>());
};

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

// clang-format off
template <class E>
concept Extents_has_rank = 
    requires (E /* exts */) { { E::rank() } -> STD_CONVERTIBLE_TO(std::size_t);
};

template <class M>
concept MDArray_type = 
    requires (M /* m */) { { M::rank() } -> STD_CONVERTIBLE_TO(std::size_t);
};
// clang-format on

} // namespace __Detail
} // namespace Sci

#endif // SCILIB_MDARRAY_SUPPORT_H
