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

namespace Sci {

namespace Mdspan = std::experimental;

//--------------------------------------------------------------------------------------------------
// Utility functions for making mdspans and mdarrays:

template <class T,
          class Extents,
          class Layout,
          class Container,
          class Accessor = Mdspan::default_accessor<T>>
constexpr Mdspan::mdspan<T, Extents, Layout, Accessor>
make_mdspan(MDArray<T, Extents, Layout, Container>& m,
            const Accessor& a = Mdspan::default_accessor<T>())
{
    return Mdspan::mdspan<T, Extents, Layout>(m.container_data(), m.mapping(), a);
}

template <class T, class Extents, class Layout, class Container = std::vector<T>, class Accessor>
constexpr MDArray<T, Extents, Layout, Container>
make_mdarray(Mdspan::mdspan<T, Extents, Layout, Accessor> m)
{
    return MDArray<T, Extents, Layout, Container>(m);
}

//--------------------------------------------------------------------------------------------------
// Equality comparisons:

template <class T, class Extents, class Layout, class Container>
constexpr bool operator==(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    using index_type = typename Extents::index_type;

    if ((a.rank() != b.rank()) || (a.extents() != b.extents())) {
        return false;
    }
    bool result = true;

    auto is_equal = [&]<class... IndexTypes>(IndexTypes... indices)
    {
#if __cpp_multidimensional_subscript
        if (a[static_cast<index_type>(std::move(indices))...] !=
            b[static_cast<index_type>(std::move(indices))...]) {
            result = false;
        }
#else
        if (a(static_cast<index_type>(std::move(indices))...) !=
            b(static_cast<index_type>(std::move(indices))...)) {
            result = false;
        }
#endif
    };
    for_each_in_extents(is_equal, a.extents(), Layout{});
    return result;
}

template <class T, class Extents, class Layout, class Container>
constexpr bool operator!=(const MDArray<T, Extents, Layout, Container>& a,
                          const MDArray<T, Extents, Layout, Container>& b)
{
    return !(a == b);
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
        std::experimental::linalg::add(a.to_mdspan(), b.to_mdspan(), res.to_mdspan());
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

    return MDArray<T, Extents, Layout, Container>(
        std::experimental::linalg::scaled(scaling_factor, v.to_mdspan()));
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator*(const T& scalar, const MDArray<T, Extents, Layout, Container>& v)
{
    using value_type = std::remove_cv_t<T>;
    value_type scaling_factor = scalar;

    return MDArray<T, Extents, Layout, Container>(
        std::experimental::linalg::scaled(scaling_factor, v.to_mdspan()));
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator/(const MDArray<T, Extents, Layout, Container>& v, const T& scalar)
{
    using value_type = std::remove_cv_t<T>;
    value_type scaling_factor = value_type{1} / scalar;

    return MDArray<T, Extents, Layout, Container>(
        std::experimental::linalg::scaled(scaling_factor, v.to_mdspan()));
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

template <class T, class Extents, class Layout, class Accessor, class Callable>
constexpr void apply(Mdspan::mdspan<T, Extents, Layout, Accessor> v, Callable&& f)
{
    using index_type = typename Extents::index_type;
    auto apply_fn = [&]<class... IndexTypes>(IndexTypes... indices)
    {
#if __cpp_multidimensional_subscript
        std::forward<Callable>(f)(v[static_cast<index_type>(std::move(indices))...]);
#else
        std::forward<Callable>(f)(v(static_cast<index_type>(std::move(indices))...));
#endif
    };
    for_each_in_extents(apply_fn, v);
}

template <class T_x,
          class Extents_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class Extents_y,
          class Layout_y,
          class Accessor_y,
          class Callable>
constexpr void apply(Mdspan::mdspan<T_x, Extents_x, Layout_x, Accessor_x> x,
                     Mdspan::mdspan<T_y, Extents_y, Layout_y, Accessor_y> y,
                     Callable&& f)
{
    using IndexType_x = typename Extents_x::index_type;
    using IndexType_y = typename Extents_y::index_type;
    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    Expects(x.rank() == y.rank());
    for (std::size_t r = 0; r < x.rank(); ++r) {
        Expects(gsl::narrow_cast<index_type>(x.extent(r)) ==
                gsl::narrow_cast<index_type>(y.extent(r)));
    }
    auto apply_fn = [&]<class... IndexTypes>(IndexTypes... indices)
    {
#if __cpp_multidimensional_subscript
        std::forward<Callable>(f)(x[static_cast<index_type>(std::move(indices))...],
                                  y[static_cast<index_type>(std::move(indices))...]);
#else
        std::forward<Callable>(f)(x(static_cast<index_type>(std::move(indices))...),
                                  y(static_cast<index_type>(std::move(indices))...));
#endif
    };
    for_each_in_extents(apply_fn, x);
}

//--------------------------------------------------------------------------------------------------
// Stream methods:

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void print(std::ostream& ostrm,
                  Mdspan::mdspan<T, Mdspan::extents<IndexType, ext>, Layout, Accessor> v)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v[i] << " ";
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
           const MDArray<T, Mdspan::extents<IndexType, ext>, Layout, Container>& v)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v[i] << " ";
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
                  Mdspan::mdspan<T, Mdspan::extents<IndexType, nrows, ncols>, Layout, Accessor> m)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
#if __cpp_multidimensional_subscript
            ostrm << std::setw(9) << m[i, j] << " ";
#else
            ostrm << std::setw(9) << m(i, j) << " ";
#endif
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
           const MDArray<T, Mdspan::extents<IndexType, nrows, ncols>, Layout, Container>& m)
{
    using index_type = IndexType;

    ostrm << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
#if __cpp_multidimensional_subscript
            ostrm << std::setw(9) << m[i, j] << " ";
#else
            ostrm << std::setw(9) << m(i, j) << " ";
#endif
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
    using extents_type = typename Matrix<T, Layout>::extents_type;
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
    auto mtmp = Matrix<T, Mdspan::layout_right>(extents_type(nr, nc), tmp);
    m = mtmp.to_mdspan();
    return istrm;
}

} // namespace Sci

#endif // SCILIB_MDARRAY_OPERATIONS_H
