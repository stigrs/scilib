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
// Modifier operations:

template <class T, class Extent, class Layout, class Accessor, class F>
constexpr void apply(stdex::mdspan<T, Extent, Layout, Accessor> v, F f)
{
    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < v.size(); ++i) {
        f(v.data_handle()[i]);
    }
}

template <class T, class Extent, class Layout, class Container, class F>
constexpr void apply(stdex::mdarray<T, Extent, Layout, Container>& m, F f)
{
    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < v.size(); ++i) {
        f(v.data()[i]);
    }
}

template <class T, class Extent, class Layout, class Accessor, class F, class U>
constexpr void apply(stdex::mdspan<T, Extent, Layout, Accessor> v, F f, const U& val)
{
    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < v.size(); ++i) {
        f(v.data_handle()[i], val);
    }
}

template <class T, class Extent, class Layout, class Container, class F, class U>
constexpr void apply(stdex::mdarray<T, Extent, Layout, Container>& m, F f, const U& val)
{
    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < v.size(); ++i) {
        f(v.data()[i], val);
    }
}

template <class T, class Extent, class Layout, class Accessor, class F>
constexpr void apply(stdex::mdspan<T, Extent, Layout, Accessor> a,
                     stdex::mdspan<T, Extent, Layout, Accessor> b,
                     F f)
{
    assert(a.size() == b.size());
    assert(a.extents() == b.extents());

    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < a.size(); ++i) {
        f(a.data()[i], b.data()[i]);
    }
}

template <class T, class Extent, class Layout, class Container, class F>
constexpr void apply(stdex::mdarray<T, Extent, Layout, Container>& a,
                     const stdex::mdarray<T, Extent, Layout, Container>& b,
                     F f)
{
    assert(a.size() == b.size());
    assert(a.extents() == b.extents());

    using index_type = typename Extent::index_type;

    for (index_type i = 0; i < a.size(); ++i) {
        f(a.data()[i], b.data()[i]);
    }
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
