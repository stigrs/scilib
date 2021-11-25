// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/mdarray_impl/vector.h>
#include <scilib/mdarray_impl/matrix.h>
#include <scilib/mdarray_impl/matrix_diag.h>
#include <scilib/mdarray_impl/type_aliases.h>
#include <scilib/linalg_impl/blas2_matrix_vector_product.h>
#include <scilib/linalg_impl/blas3_matrix_product.h>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace Scilib {

//------------------------------------------------------------------------------
// Vector operations:

template <typename T>
inline bool operator==(const Vector<T>& a, const Vector<T>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
inline bool operator!=(const Vector<T>& a, const Vector<T>& b)
{
    return !(a == b);
}

template <typename T>
inline bool operator<(const Vector<T>& a, const Vector<T>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <typename T>
inline bool operator>(const Vector<T>& a, const Vector<T>& b)
{
    return b < a;
}

template <typename T>
inline bool operator<=(const Vector<T>& a, const Vector<T>& b)
{
    return !(a > b);
}

template <typename T>
inline bool operator>=(const Vector<T>& a, const Vector<T>& b)
{
    return !(a < b);
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& v)
{
    Vector<T> res = v;
    return res *= -T{1};
}

template <typename T>
inline Vector<T> operator+(const Vector<T>& a, const Vector<T>& b)
{
    assert(a.size() == b.size());
    Vector<T> res = a;
    return res += b;
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& a, const Vector<T>& b)
{
    assert(a.size() == b.size());
    Vector<T> res = a;
    return res -= b;
}

template <typename T>
inline Vector<T> operator+(const Vector<T>& v, const T& scalar)
{
    Vector<T> res = v;
    return res += scalar;
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& v, const T& scalar)
{
    Vector<T> res = v;
    return res -= scalar;
}

template <typename T>
inline Vector<T> operator*(const Vector<T>& v, const T& scalar)
{
    Vector<T> res = v;
    return res *= scalar;
}

template <typename T>
inline Vector<T> operator*(const T& scalar, const Vector<T>& v)
{
    Vector<T> res = v;
    return res *= scalar;
}

template <typename T>
inline Vector<T> operator/(const Vector<T>& v, const T& scalar)
{
    Vector<T> res = v;
    return res /= scalar;
}

template <typename T>
inline Vector<T> operator%(const Vector<T>& v, const T& scalar)
{
    Vector<T> res = v;
    return res %= scalar;
}

//------------------------------------------------------------------------------
// Matrix operations:

template <typename T>
inline bool operator==(const Matrix<T>& a, const Matrix<T>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
inline bool operator!=(const Matrix<T>& a, const Matrix<T>& b)
{
    return !(a == b);
}

template <typename T>
inline bool operator<(const Matrix<T>& a, const Matrix<T>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <typename T>
inline bool operator>(const Matrix<T>& a, const Matrix<T>& b)
{
    return b < a;
}

template <typename T>
inline bool operator<=(const Matrix<T>& a, const Matrix<T>& b)
{
    return !(a > b);
}

template <typename T>
inline bool operator>=(const Matrix<T>& a, const Matrix<T>& b)
{
    return !(a < b);
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& m)
{
    Matrix<T> res = m;
    return res *= -T{1};
}

template <typename T>
inline Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b)
{
    assert(a.rows() == b.rows());
    assert(a.cols() == b.cols());
    Matrix<T> res = a;
    return res += b;
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b)
{
    assert(a.rows() == b.rows());
    assert(a.cols() == b.cols());
    Matrix<T> res = a;
    return res -= b;
}

template <typename T>
inline Matrix<T> operator+(const Matrix<T>& m, const T& scalar)
{
    Matrix<T> res = m;
    return res += scalar;
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& m, const T& scalar)
{
    Matrix<T> res = m;
    return res -= scalar;
}

template <typename T>
inline Matrix<T> operator*(const Matrix<T>& m, const T& scalar)
{
    Matrix<T> res = m;
    return res *= scalar;
}

template <typename T>
inline Matrix<T> operator*(const T& scalar, const Matrix<T>& m)
{
    Matrix<T> res = m;
    return res *= scalar;
}

template <typename T>
inline Matrix<T> operator/(const Matrix<T>& m, const T& scalar)
{
    Matrix<T> res = m;
    return res /= scalar;
}

template <typename T>
inline Matrix<T> operator%(const Matrix<T>& m, const T& scalar)
{
    Matrix<T> res = m;
    return res %= scalar;
}

//------------------------------------------------------------------------------
// Matrix-matrix product:

template <typename T>
inline Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b)
{
    return Scilib::Linalg::matrix_matrix_product(a.view(), b.view());
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

template <class T, class F>
inline void apply(Vector_view<T> v, F f)
{
    for (std::size_t i = 0; i < v.extent(0); ++i) {
        f(v(i));
    }
}

template <class T, class F>
inline void apply(Subvector_view<T> v, F f)
{
    for (std::size_t i = 0; i < v.extent(0); ++i) {
        f(v(i));
    }
}

template <class T, class F>
inline void apply(Matrix_view<T> m, F f)
{
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
            f(m(i, j));
        }
    }
}

template <class T, class F>
inline void apply(Submatrix_view<T> m, F f)
{
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
            f(m(i, j));
        }
    }
}

//------------------------------------------------------------------------------
// Stream methods:

template <typename T>
inline void print(std::ostream& ostrm, Vector_view<T> v)
{
    ostrm << v.size() << '\n' << '{';
    for (std::size_t i = 0; i < v.size(); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.size() - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
}

template <typename T>
inline void print(std::ostream& ostrm, Subvector_view<T> v)
{
    ostrm << v.size() << '\n' << '{';
    for (std::size_t i = 0; i < v.size(); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.size() - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << "}\n";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& ostrm, const Vector<T>& v)
{
    ostrm << v.size() << '\n' << '{';
    for (std::size_t i = 0; i < v.size(); ++i) {
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
    std::size_t n;
    istrm >> n;
    std::vector<T> tmp(n);

    char ch;
    istrm >> ch; // {
    for (std::size_t i = 0; i < n; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    v = Vector<T>(tmp);
    return istrm;
}

template <typename T>
inline void print(std::ostream& ostrm, Matrix_view<T> m)
{
    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.extent(0) - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << "}\n";
}

template <typename T>
inline void print(std::ostream& ostrm, Submatrix_view<T> m)
{
    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
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
    ostrm << m.extent(0) << " x " << m.extent(1) << '\n' << '{';
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
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
    std::size_t nr;
    std::size_t nc;
    char ch;

    istrm >> nr >> ch >> nc;
    std::vector<T> tmp(nr * nc);

    istrm >> ch; // {
    for (std::size_t i = 0; i < nr * nc; ++i) {
        istrm >> tmp[i];
    }
    istrm >> ch; // }
    m = Matrix<T>(tmp, nr, nc);
    return istrm;
}

} // namespace Scilib
