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
    MDArray<T, Extents, Layout, Container> res = v;
    return res *= scalar;
}

template <class T, class Extents, class Layout, class Container>
constexpr MDArray<T, Extents, Layout, Container>
operator*(const T& scalar, const MDArray<T, Extents, Layout, Container>& v)
{
    MDArray<T, Extents, Layout, Container> res = v;
    return res *= scalar;
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

template <class T, class Layout, class Container>
constexpr Matrix<T, Layout, Container> operator*(const Matrix<T, Layout, Container>& a,
                                                 const Matrix<T, Layout, Container>& b)
{
    return Sci::Linalg::matrix_product(a, b);
}

//--------------------------------------------------------------------------------------------------
// Matrix-vector product:

template <class T, class Layout, class Container>
constexpr Vector<T, Layout, Container> operator*(const Matrix<T, Layout, Container>& a,
                                                 const Vector<T, Layout, Container>& x)
{
    return Sci::Linalg::matrix_vector_product(a, x);
}

//--------------------------------------------------------------------------------------------------
// Apply operations:

template <class T, std::size_t ext, class Layout, class Accessor, class F>
constexpr void apply(stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> v, F f)
{
    using index_type = index;

    for (index_type i = 0; i < v.extent(0); ++i) {
        f(v(i));
    }
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor, class F>
constexpr void apply(stdex::mdspan<T, stdex::extents<index, nrows, ncols>, Layout, Accessor> m, F f)
{
    using index_type = index;

    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            f(m(i, j));
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

    ostrm << v.extent(0) << '\n' << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
}

template <class T, class Layout, class Container>
inline std::ostream& operator<<(std::ostream& ostrm, const Vector<T, Layout, Container>& v)
{
    using index_type = typename Vector<T, Layout, Container>::index_type;

    ostrm << v.size() << '\n' << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
    return ostrm;
}

template <class T, class Layout, class Container>
inline std::istream& operator>>(std::istream& istrm, Vector<T, Layout, Container>& v)
{
    using index_type = typename Vector<T, Layout, Container>::index_type;

    index_type n;
    istrm >> n;
    std::vector<T> tmp(n);

    char ch;
    istrm >> ch; // {
    for (index_type i = 0; i < n; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    v = Vector<T, Layout, Container>(tmp, tmp.size());
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

    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << "}\n";
}

template <class T, class Layout, class Container>
inline std::ostream& operator<<(std::ostream& ostrm, const Matrix<T, Layout, Container>& m)
{
    using index_type = typename Matrix<T, Layout, Container>::index_type;

    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (index_type i = 0; i < m.extent(0); ++i) {
        for (index_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << "}\n";
    return ostrm;
}

template <class T, class Layout, class Container>
inline std::istream& operator>>(std::istream& istrm, Matrix<T, Layout, Container>& m)
{
    using index_type = typename Matrix<T, Layout, Container>::index_type;

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
    auto mtmp = Matrix<T, stdex::layout_right, Container>(tmp, nr, nc);
    m = mtmp.view();
    return istrm;
}

} // namespace Sci

#endif // SCILIB_MDARRAY_OPERATIONS_H
