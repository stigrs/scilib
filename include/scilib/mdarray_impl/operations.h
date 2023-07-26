// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_OPERATIONS_H
#define SCILIB_MDARRAY_OPERATIONS_H

#include "../linalg_impl/blas1_add.h"
#include "../linalg_impl/blas1_scale.h"
#include "../linalg_impl/blas2_matrix_vector_product.h"
#include "../linalg_impl/blas3_matrix_product.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <utility>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/indices.hpp>

namespace Sci {

namespace stdex = std::experimental;

//--------------------------------------------------------------------------------------------------
// Utility functions for making mdspans and mdarrays:

template <class T,
          class Extents,
          class Layout,
          class Container,
          class Accessor = stdex::default_accessor<T>>
constexpr stdex::mdspan<T, Extents, Layout, Accessor>
make_mdspan(MDArray<T, Extents, Layout, Container>& m,
            const Accessor& a = stdex::default_accessor<T>())
{
    return stdex::mdspan<T, Extents, Layout>(m.data(), m.mapping(), a);
}

template <class T, class Extents, class Layout, class Container = std::vector<T>, class Accessor>
constexpr MDArray<T, Extents, Layout, Container>
make_mdarray(stdex::mdspan<T, Extents, Layout, Accessor> m)
{
    return MDArray<T, Extents, Layout, Container>(m);
}

//--------------------------------------------------------------------------------------------------
// For each extents:
//
// Copyright (2022) National Technology & Engineering Solutions of Sandia, LLC (NTESS).
// See https://kokkos.org/LICENSE for license information.

namespace __Detail {

template <std::size_t... Lefts, std::size_t... Rights>
auto concat_index_sequence(std::index_sequence<Lefts...>, std::index_sequence<Rights...>)
{
    return std::index_sequence<Lefts..., Rights...>{};
}

auto reverse_index_sequence(std::index_sequence<> x) { return x; }

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
    [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
        auto v = ranges::views::cartesian_product(
            ranges::views::indices(IndexType(0), e.extent(Indices))...);
        for (const auto& tuple_of_indices : v) {
            // In the layout_left case, we undo the reversal of each tuple
            // by getting its elements in reverse order.
            [&]<std::size_t... InnerIndices>(std::index_sequence<InnerIndices...>) {
                std::forward<Callable>(f)(std::get<InnerIndices>(tuple_of_indices)...);
            }(rank_sequence);
        }
    }(rank_sequence);
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
    Layout layout_type; 
    for_each_in_extents(f, m.extents(), layout_type);
}

template <class Callable, class T, class Extents, class Layout, class Container>
void for_each_in_extents(Callable&& f, MDArray<T, Extents, Layout, Container>& m)
{
    for_each_in_extents(f, m.view());
}

//--------------------------------------------------------------------------------------------------
// Equality comparisons:

template <class T, class Extents, class Layout, class Container>
constexpr bool operator==(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator!=(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    bool result = true;
    if (a.extents() == b.extents()) {
        result = !(a == b);
    }
    return result;
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator<(const MDArray<T, Extents, Layout, Container>& a,
                         const MDArray<T, Extents, Layout, Container>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator>(const MDArray<T, Extents, Layout, Container>& a,
                         const MDArray<T, Extents, Layout, Container>& b)
{
    return b < a;
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator<=(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    return !(a > b);
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator>=(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    return !(a < b);
}

//--------------------------------------------------------------------------------------------------
// Arithmetic operations:

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator-(const MDArray<T, Extents, Layout, Container>& v)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res *= -T{1};
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator+(const MDArray<T, Extents, Layout, Container>& a,
          const MDArray<T, Extents, Layout, Container>& b)
{
    if constexpr (Extents::rank() <= 1) {
        MDArray<T, Extents, Layout, Container> res(a.extents());
        std::experimental::linalg::add(a.view(), b.view(), res.view());
        return res;
    }
    else {
        MDArray<T, Extents, Layout, Container> res = a;
        return res += b;
    }
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator-(const MDArray<T, Extents, Layout, Container>& a,
          const MDArray<T, Extents, Layout, Container>& b)
{
    MDArray<T, Extents, Layout, Container> res = a;
    return res -= b;
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator+(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res += scalar;
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator-(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res -= scalar;
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator*(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    using value_type = std::remove_cv_t<T>;
    value_type scaling_factor = scalar;

    if constexpr (Extents::rank() <= 7) {
        return MDArray<T, Extents, Layout, Container>(
            std::experimental::linalg::scaled(scaling_factor, v.view()));
    }
    else {
        MDArray<T, Extents, Layout, Container> res = v;
        return res *= scalar;
    }
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator*(const T& scalar, const MDArray<T, Extents, Layout, Container>& v)
{
    using value_type = std::remove_cv_t<T>;
    value_type scaling_factor = scalar;

    if constexpr (Extents::rank() <= 7) {
        return MDArray<T, Extents, Layout, Container>(
            std::experimental::linalg::scaled(scaling_factor, v.view()));
    }
    else {
        MDArray<T, Extents, Layout, Container> res = v;
        return res *= scalar;
    }
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator/(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res /= scalar;
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator%(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res %= scalar;
}

//--------------------------------------------------------------------------------------------------
// Matrix-matrix product:

template <class T, class Layout>
constexpr Matrix<T, Layout> operator*(const Matrix<T, Layout>& a, const Matrix<T, Layout>& b)
{
    return Sci::Linalg::matrix_product(a, b);
}

//--------------------------------------------------------------------------------------------------
// Matrix-vector product:

template <class T, class Layout>
constexpr Vector<T, Layout> operator*(const Matrix<T, Layout>& a, const Vector<T, Layout>& x)
{
    return Sci::Linalg::matrix_vector_product(a, x);
}

//--------------------------------------------------------------------------------------------------
// Apply operations:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor, class F>
    requires(std::is_integral_v<IndexType>)
constexpr void apply(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v, F f)
{
    using index_type = IndexType;

    for (index_type i = 0; i < v.extent(0); ++i) {
        f(v(i));
    }
}

template <class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y,
          class F>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
constexpr void apply(stdex::mdspan<T_x, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
                     stdex::mdspan<T_y, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
                     F f)
{
    Expects(x.extent(0) == x.extent(1));
    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (index_type i = 0; i < x.extent(0); ++i) {
        f(x(i), y(i));
    }
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor,
          class F>
    requires(std::is_integral_v<IndexType>)
constexpr void apply(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> m,
                     F f)
{
    using index_type = IndexType;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            for (index_type i = 0; i < m.extent(0); ++i) {
                f(m(i, j));
            }
        }
    }
    else {
        for (index_type i = 0; i < m.extent(0); ++i) {
            for (index_type j = 0; j < m.extent(1); ++j) {
                f(m(i, j));
            }
        }
    }
}

template <class T_a,
          class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class T_b,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Layout_b,
          class Accessor_b,
          class F>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>)
constexpr void
apply(stdex::mdspan<T_a, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
      stdex::mdspan<T_b, stdex::extents<IndexType_b, nrows_b, ncols_b>, Layout_b, Accessor_b> b,
      F f)
{
    Expects(a.extent(0) == b.extent(0));
    Expects(a.extent(1) == b.extent(1));

    using index_type = std::common_type_t<IndexType_a, IndexType_b>;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; j < a.extent(1); ++j) {
            f(a(i, j), b(i, j));
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Stream methods:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void print(std::ostream& ostrm,
                  stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << '}';
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline std::ostream&
operator<<(std::ostream& ostrm,
           const MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << '}';
    return ostrm;
}

template <class T, class Layout>
inline std::istream& operator>>(std::istream& istrm, Vector<T, Layout>& v)
{
    using index_type = typename Vector<T, Layout>::index_type;

    index_type n;
    istrm >> n;
    std::vector<T> tmp(n);

    char ch;
    istrm >> ch; // {
    for (index_type i = 0; i < n; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    v = Vector<T, Layout>(tmp, tmp.size());
    return istrm;
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void print(std::ostream& ostrm,
                  stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> m)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << '}';
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline std::ostream&
operator<<(std::ostream& ostrm,
           const MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& m)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << '}';
    return ostrm;
}

template <class T, class Layout>
inline std::istream& operator>>(std::istream& istrm, Matrix<T, Layout>& m)
{
    using index_type = typename Matrix<T, Layout>::index_type;

    index_type nr;
    index_type nc;
    char ch;

    istrm >> nr >> ch >> nc;
    std::vector<T> tmp(nr * nc);

    istrm >> ch; // {
    for (index_type i = 0; i < nr * nc; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    auto mtmp = Matrix<T, stdex::layout_right>(tmp, nr, nc);
    m = mtmp.view();
    return istrm;
}

} // namespace Sci

#endif // SCILIB_MDARRAY_OPERATIONS_H
