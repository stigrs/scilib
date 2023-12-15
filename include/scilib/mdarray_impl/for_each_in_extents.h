// Copyright (c) 2023 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_FOR_EACH_IN_EXTENTS_H
#define SCILIB_MDARRAY_FOR_EACH_IN_EXTENTS_H

#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <type_traits>

//--------------------------------------------------------------------------------------------------
// For each in extents:
//
// Copyright (2022) National Technology & Engineering Solutions of Sandia, LLC (NTESS).
// See https://kokkos.org/LICENSE for license information.

namespace Sci {

namespace __Detail {

template <std::size_t... Lefts, std::size_t... Rights>
auto concat_index_sequence(std::index_sequence<Lefts...>, std::index_sequence<Rights...>)
{
    return std::index_sequence<Lefts..., Rights...>{};
}

inline auto reverse_index_sequence(std::index_sequence<> x) { return x; }

template <std::size_t First, std::size_t... Rest>
auto reverse_index_sequence(std::index_sequence<First, Rest...>)
{
    return concat_index_sequence(reverse_index_sequence(std::index_sequence<Rest...>{}),
                                 std::index_sequence<First>{});
}

template <std::size_t N>
auto make_reverse_index_sequence()
{
    return reverse_index_sequence(std::make_index_sequence<N>());
}

template <class Callable, class IndexType, std::size_t... Extents, std::size_t... RankIndices>
void for_each_in_extents_impl(Callable&& f,
                              stdex::extents<IndexType, Extents...> e,
                              std::index_sequence<RankIndices...> rank_sequence)
{
    // In the layout_left case, caller passes in N-1, N-2, ..., 1, 0.
    // This reverses the order of the Cartesian product,
    // but also reverses the order of indices in each tuple.
    [&]<std::size_t... Indices>(std::index_sequence<Indices...>)
    {
        auto v = ranges::views::cartesian_product(
            ranges::views::iota(IndexType(0), e.extent(Indices))...);
        for (const auto& tuple_of_indices : v) {
            // In the layout_left case, we undo the reversal of each tuple
            // by getting its elements in reverse order.
            [&]<std::size_t... InnerIndices>(std::index_sequence<InnerIndices...>)
            {
                std::forward<Callable>(f)(std::get<InnerIndices>(tuple_of_indices)...);
            }
            (rank_sequence);
        }
    }
    (rank_sequence);
}
} // namespace __Detail

template <class Callable, class IndexType, std::size_t... Extents, class Layout>
void for_each_in_extents(Callable&& f, stdex::extents<IndexType, Extents...> e, Layout)
{
    using layout_type = std::remove_cvref_t<Layout>;
    if constexpr (std::is_same_v<layout_type, stdex::layout_left>) {
        __Detail::for_each_in_extents_impl(std::forward<Callable>(f), e,
                                           __Detail::make_reverse_index_sequence<e.rank()>());
    }
    else { // layout_right or any other layout
        __Detail::for_each_in_extents_impl(std::forward<Callable>(f), e,
                                           std::make_index_sequence<e.rank()>());
    }
}

template <class Callable,
          class ElementType,
          class IndexType,
          std::size_t... Extents,
          class Layout,
          class Accessor>
void for_each_in_extents(
    Callable&& f,
    stdex::mdspan<ElementType, stdex::extents<IndexType, Extents...>, Layout, Accessor> m)
{
    for_each_in_extents(f, m.extents(), Layout{});
}

template <class Callable, class T, class Extents, class Layout, class Container>
void for_each_in_extents(Callable&& f, MDArray<T, Extents, Layout, Container>& m)
{
    for_each_in_extents(f, m.to_mdspan());
}

} // namespace Sci

#endif // SCILIB_MDARRAY_FOR_EACH_IN_EXTENTS_H
