// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SUPPORT_H
#define SCILIB_MDARRAY_SUPPORT_H

#include <array>
#include <type_traits>

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

} // namespace __Detail
} // namespace Sci

#endif // SCILIB_MDARRAY_SUPPORT_H
