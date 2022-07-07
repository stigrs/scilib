// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_OPERATIONS_H
#define SCILIB_MDARRAY_OPERATIONS_H

#include "../linalg_impl/blas2_matrix_vector_product.h"
#include "../linalg_impl/blas3_matrix_product.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <type_traits>

namespace Sci {

namespace stdex = std::experimental;

//--------------------------------------------------------------------------------------------------
// Properties:

template <class T, class Extents, class Layout, class Allocator>
constexpr std::size_t rank(const MDArray<T, Extents, Layout, Allocator>& m)
{
    return m.rank();
}

template <class T, class Extents, class Layout, class Accessor>
constexpr std::size_t rank(stdex::mdspan<T, Extents, Layout, Accessor> m)
{
    return m.rank();
}

template <class T, class Extents, class Layout, class Allocator>
constexpr std::ptrdiff_t ssize(const MDArray<T, Extents, Layout, Allocator>& m)
{
    return m.ssize();
}

template <class T, class Extents, class Layout, class Accessor>
constexpr std::ptrdiff_t ssize(stdex::mdspan<T, Extents, Layout, Accessor> m)
{
    return static_cast<std::ptrdiff_t>(m.size());
}

template <class T, class Extents, class Layout, class Allocator>
constexpr std::ptrdiff_t sextent(const MDArray<T, Extents, Layout, Allocator>& m, std::size_t dim)
{
    return m.sextent(dim);
}

template <class T, class Extents, class Layout, class Accessor>
constexpr std::ptrdiff_t sextent(stdex::mdspan<T, Extents, Layout, Accessor> m, std::size_t dim)
{
    return static_cast<std::ptrdiff_t>(m.extent(dim));
}

//--------------------------------------------------------------------------------------------------
// Equality comparisons:

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator==(const MDArray<T, Extents, Layout, Allocator>& a,
                          const MDArray<T, Extents, Layout, Allocator>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator!=(const MDArray<T, Extents, Layout, Allocator>& a,
                          const MDArray<T, Extents, Layout, Allocator>& b)
{
    bool result = true;
    if (a.view().extents() == b.view().extents()) {
        result = !(a == b);
    }
    return result;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator<(const MDArray<T, Extents, Layout, Allocator>& a,
                         const MDArray<T, Extents, Layout, Allocator>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator>(const MDArray<T, Extents, Layout, Allocator>& a,
                         const MDArray<T, Extents, Layout, Allocator>& b)
{
    return b < a;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator<=(const MDArray<T, Extents, Layout, Allocator>& a,
                          const MDArray<T, Extents, Layout, Allocator>& b)
{
    return !(a > b);
}

template <class T, class Extents, class Layout, class Allocator>
constexpr bool operator>=(const MDArray<T, Extents, Layout, Allocator>& a,
                          const MDArray<T, Extents, Layout, Allocator>& b)
{
    return !(a < b);
}

//--------------------------------------------------------------------------------------------------
// Arithmetic operations:

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator-(const MDArray<T, Extents, Layout, Allocator>& v)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res *= -T{1};
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator+(const MDArray<T, Extents, Layout, Allocator>& a,
          const MDArray<T, Extents, Layout, Allocator>& b)
{
    MDArray<T, Extents, Layout, Allocator> res = a;
    return res += b;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator-(const MDArray<T, Extents, Layout, Allocator>& a,
          const MDArray<T, Extents, Layout, Allocator>& b)
{
    MDArray<T, Extents, Layout, Allocator> res = a;
    return res -= b;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator+(const MDArray<T, Extents, Layout, Allocator>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res += scalar;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator-(const MDArray<T, Extents, Layout, Allocator>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res -= scalar;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator*(const MDArray<T, Extents, Layout, Allocator>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res *= scalar;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator*(const T& scalar, const MDArray<T, Extents, Layout, Allocator>& v)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res *= scalar;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator/(const MDArray<T, Extents, Layout, Allocator>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res /= scalar;
}

template <class T, class Extents, class Layout, class Allocator>
constexpr MDArray<T, Extents, Layout, Allocator>
operator%(const MDArray<T, Extents, Layout, Allocator>& v, const T& scalar)
{
    MDArray<T, Extents, Layout, Allocator> res = v;
    return res %= scalar;
}

//--------------------------------------------------------------------------------------------------
// Matrix-matrix product:

template <class T, class Layout, class Allocator>
constexpr Matrix<T, Layout, Allocator> operator*(const Matrix<T, Layout, Allocator>& a,
                                                 const Matrix<T, Layout, Allocator>& b)
{
    return Sci::Linalg::matrix_product(a, b);
}

//--------------------------------------------------------------------------------------------------
// Matrix-vector product:

template <class T, class Layout, class Allocator>
constexpr Vector<T, Layout, Allocator> operator*(const Matrix<T, Layout, Allocator>& a,
                                                 const Vector<T, Layout, Allocator>& x)
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

template <class T, std::size_t ext, class Layout, class Accessor>
inline void print(std::ostream& ostrm,
                  stdex::mdspan<T, stdex::extents<index, ext>, Layout, Accessor> v)
{
    using index_type = index;

    ostrm << v.extent(0) << '\n' << '{';
    for (index_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
}

template <class T, class Layout, class Allocator>
inline std::ostream& operator<<(std::ostream& ostrm, const Vector<T, Layout, Allocator>& v)
{
    using index_type = typename Vector<T, Layout, Allocator>::index_type;

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

template <class T, class Layout, class Allocator>
inline std::istream& operator>>(std::istream& istrm, Vector<T, Layout, Allocator>& v)
{
    using index_type = typename Vector<T, Layout, Allocator>::index_type;

    index_type n;
    istrm >> n;
    std::vector<T> tmp(n);

    char ch;
    istrm >> ch; // {
    for (index_type i = 0; i < n; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    v = Vector<T, Layout, Allocator>(tmp, tmp.size());
    return istrm;
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
inline void print(std::ostream& ostrm,
                  stdex::mdspan<T, stdex::extents<index, nrows, ncols>, Layout, Accessor> m)
{
    using index_type = index;

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

template <class T, class Layout, class Allocator>
inline std::ostream& operator<<(std::ostream& ostrm, const Matrix<T, Layout, Allocator>& m)
{
    using index_type = typename Matrix<T, Layout, Allocator>::index_type;

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

template <class T, class Layout, class Allocator>
inline std::istream& operator>>(std::istream& istrm, Matrix<T, Layout, Allocator>& m)
{
    using index_type = typename Matrix<T, Layout, Allocator>::index_type;

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
    auto mtmp = Matrix<T, stdex::layout_right, Allocator>(tmp, nr, nc);
    m = mtmp.view();
    return istrm;
}

} // namespace Sci

#endif // SCILIB_MDARRAY_OPERATIONS_H
