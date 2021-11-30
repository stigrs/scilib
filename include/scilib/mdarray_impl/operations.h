// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_OPERATIONS_H
#define SCILIB_MDARRAY_OPERATIONS_H

#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/slice.h>
#include <scilib/linalg_impl/blas2_matrix_vector_product.h>
#include <scilib/linalg_impl/blas3_matrix_product.h>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace Scilib {

namespace stdex = std::experimental;

//------------------------------------------------------------------------------
// Equality comparisons:

template <class T, class Extents>
inline bool operator==(const MDArray<T, Extents>& a,
                       const MDArray<T, Extents>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

template <class T, class Extents>
inline bool operator!=(const MDArray<T, Extents>& a,
                       const MDArray<T, Extents>& b)
{
    return !(a == b);
}

template <class T, class Extents>
inline bool operator<(const MDArray<T, Extents>& a,
                      const MDArray<T, Extents>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <class T, class Extents>
inline bool operator>(const MDArray<T, Extents>& a,
                      const MDArray<T, Extents>& b)
{
    return b < a;
}

template <class T, class Extents>
inline bool operator<=(const MDArray<T, Extents>& a,
                       const MDArray<T, Extents>& b)
{
    return !(a > b);
}

template <class T, class Extents>
inline bool operator>=(const MDArray<T, Extents>& a,
                       const MDArray<T, Extents>& b)
{
    return !(a < b);
}

//------------------------------------------------------------------------------
// Arithmetic operations:

template <class T, class Extents>
inline MDArray<T, Extents> operator-(const MDArray<T, Extents>& v)
{
    MDArray<T, Extents> res = v;
    return res *= -T{1};
}

template <class T, class Extents>
inline MDArray<T, Extents> operator+(const MDArray<T, Extents>& a,
                                     const MDArray<T, Extents>& b)
{
    MDArray<T, Extents> res = a;
    return res += b;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator-(const MDArray<T, Extents>& a,
                                     const MDArray<T, Extents>& b)
{
    MDArray<T, Extents> res = a;
    return res -= b;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator+(const MDArray<T, Extents>& v,
                                     const T& scalar)
{
    MDArray<T, Extents> res = v;
    return res += scalar;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator-(const MDArray<T, Extents>& v,
                                     const T& scalar)
{
    MDArray<T, Extents> res = v;
    return res -= scalar;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator*(const MDArray<T, Extents>& v,
                                     const T& scalar)
{
    MDArray<T, Extents> res = v;
    return res *= scalar;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator*(const T& scalar,
                                     const MDArray<T, Extents>& v)
{
    MDArray<T, Extents> res = v;
    return res *= scalar;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator/(const MDArray<T, Extents>& v,
                                     const T& scalar)
{
    MDArray<T, Extents> res = v;
    return res /= scalar;
}

template <class T, class Extents>
inline MDArray<T, Extents> operator%(const MDArray<T, Extents>& v,
                                     const T& scalar)
{
    MDArray<T, Extents> res = v;
    return res %= scalar;
}

//------------------------------------------------------------------------------
// Matrix-matrix product:

template <typename T>
inline Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b)
{
    return Scilib::Linalg::matrix_product(a.view(), b.view());
}

//------------------------------------------------------------------------------
// Matrix-vector product:

template <typename T>
inline Vector<T> operator*(const Matrix<T>& a, const Vector<T>& x)
{
    return Scilib::Linalg::matrix_vector_product(a.view(), x.view());
}

//------------------------------------------------------------------------------
// Apply operations:

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor,
          class F>
// clang-format off
inline void 
apply(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v, F f)
// clang-format on
{
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < v.extent(0); ++i) {
        f(v(i));
    }
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor,
          class F>
inline void
apply(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m, F f)
{
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < m.extent(0); ++i) {
        for (size_type j = 0; j < m.extent(1); ++j) {
            f(m(i, j));
        }
    }
}

//------------------------------------------------------------------------------
// Stream methods:

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline void print(std::ostream& ostrm,
                  stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    using size_type = stdex::extents<>::size_type;

    ostrm << v.extent(0) << '\n' << '{';
    for (size_type i = 0; i < v.extent(0); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.extent(0) - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& ostrm, const Vector<T>& v)
{
    using size_type = stdex::extents<>::size_type;

    ostrm << v.size() << '\n' << '{';
    for (size_type i = 0; i < v.size(); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.size() - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
    return ostrm;
}

template <typename T>
inline std::istream& operator>>(std::istream& istrm, Vector<T>& v)
{
    using size_type = stdex::extents<>::size_type;

    size_type n;
    istrm >> n;
    std::vector<T> tmp(n);

    char ch;
    istrm >> ch; // {
    for (size_type i = 0; i < n; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    v = Vector<T>(tmp);
    return istrm;
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline void
print(std::ostream& ostrm,
      stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m)
{
    using size_type = stdex::extents<>::size_type;

    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (size_type i = 0; i < m.extent(0); ++i) {
        for (size_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << "}\n";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& ostrm, const Matrix<T>& m)
{
    using size_type = stdex::extents<>::size_type;

    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (size_type i = 0; i < m.extent(0); ++i) {
        for (size_type j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << "}\n";
    return ostrm;
}

template <typename T>
inline std::istream& operator>>(std::istream& istrm, Matrix<T>& m)
{
    using size_type = stdex::extents<>::size_type;

    size_type nr;
    size_type nc;
    char ch;

    istrm >> nr >> ch >> nc;
    std::vector<T> tmp(nr * nc);

    istrm >> ch; // {
    for (size_type i = 0; i < nr * nc; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    m = Matrix<T>(tmp, nr, nc);
    return istrm;
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_OPERATIONS_H
