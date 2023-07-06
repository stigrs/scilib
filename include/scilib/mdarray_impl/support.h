// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SUPPORT_H
#define SCILIB_MDARRAY_SUPPORT_H

#include <array>
#include <experimental/mdspan>
#include <gsl/gsl>
#include <initializer_list>

namespace Sci {

namespace __Detail {

namespace stdex = std::experimental;

//--------------------------------------------------------------------------------------------------
// Type traits:

template <class E>
struct Is_extents : std::false_type {
};

template <class IndexType, std::size_t... SizeTypes>
struct Is_extents<std::experimental::extents<IndexType, SizeTypes...>> : std::true_type {
};

template <class E>
static constexpr bool Is_extents_v = Is_extents<E>::value;

//--------------------------------------------------------------------------------------------------
// mdspan returns m.extents() by reference, create a copy:

template <class T, class Extents, class Layout, class Accessor>
inline Extents extents(stdex::mdspan<T, Extents, Layout, Accessor> m)
{
    // mdspan returns m.extents() by reference, need to make a copy
    using index_type = typename Extents::index_type;

    std::array<index_type, Extents::rank()> res;
    for (std::size_t i = 0; i < m.rank(); ++i) {
        res[i] = m.extent(i);
    }
    return Extents(res);
}

//--------------------------------------------------------------------------------------------------
// MDArray list initialization:
//
// Based on the N-Dimensional Matrix Design by Andrew Sutton and Bjarne Stroustrup available at
//   https://code.google.com/archive/p/origin/
// under a MIT license.

// Describes the structure of a nested std::initializer_list.
template <class T, std::size_t N>
struct MDArray_init {
    using type = std::initializer_list<typename MDArray_init<T, N - 1>::type>;
};

// The N == 1 is special; that is were we go to the (most deeply nested)
// std::initializer_list<T>.
template <class T>
struct MDArray_init<T, 1> {
    using type = std::initializer_list<T>;
};

// To avoid surprises, N == 0 is defined to be an error.
template <class T>
struct MDArray_init<T, 0>;

// MDArray initializer.
template <class T, std::size_t N>
using MDArray_initializer = typename MDArray_init<T, N>::type;

template <std::size_t N, class List>
inline bool check_non_jagged(const List& list);

template <std::size_t N, class I, class List>
    requires(N == 1)
inline void add_extents(I& first, const List& list) { *first = list.size(); }

// Recursion through nested std::initializer_list.
template <std::size_t N, class I, class List>
    requires(N > 1)
inline void add_extents(I& first, const List& list)
{
    Expects(check_non_jagged<N>(list));
    *first = list.size(); // store this size
    add_extents<N - 1>(++first, *list.begin());
}

// Determine the shape of the MDArray:
//   + Checks that the tree is really N deep
//   + Checks that each row has the same number of elements
//   + Sets the extent of each row
//
template <std::size_t N, class List>
inline std::array<std::size_t, N> derive_extents(const List& list)
{
    std::array<std::size_t, N> exts;
    auto f = exts.begin();
    add_extents<N>(f, list); // add sizes (extents) to exts
    return exts;
}

// Check that all rows have the same number of elements.
template <std::size_t N, class List>
inline bool check_non_jagged(const List& list)
{
    auto i = list.begin();
    for (auto j = i + 1; j != list.end(); ++j) {
        if (derive_extents<N - 1>(*i) != derive_extents<N - 1>(*j)) {
            return false;
        }
    }
    return true;
}

// When we reach a list with non-initializer_list elements, we insert
// those elements into our container.
template <class T, class Container>
inline void add_list(const T* first, const T* last, Container& ctr)
{
    ctr.insert(ctr.end(), first, last);
}

template <class T, class Container>
inline void add_list(const std::initializer_list<T>* first,
                     const std::initializer_list<T>* last,
                     Container& ctr)
{
    while (first != last) {
        add_list(first->begin(), first->end(), ctr);
        ++first;
    }
}

// Copy elements of the tree of std::initializer_list to a container.
template <class T, class Container>
inline void insert_flat(std::initializer_list<T> list, Container& ctr)
{
    add_list(list.begin(), list.end(), ctr);
}

} // namespace __Detail
} // namespace Sci

#endif // SCILIB_MDARRAY_SUPPORT_H
