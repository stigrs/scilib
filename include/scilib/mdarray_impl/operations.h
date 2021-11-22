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
#include <algorithm>
#include <iostream>

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

template <typename T>
std::ostream& operator<<(std::ostream& ostrm, const Vector<T>& v)
{
    ostrm << v.size() << '\n' << "{ ";
    for (std::size_t i = 0; i < v.size(); ++i) {
        ostrm << std::setw(9) << v(i) << " ";
        if (!((i + 1) % 7) && (i != (v.size() - 1))) {
            ostrm << "\n  ";
        }
    }
    ostrm << '}';
    return ostrm;
}

template <typename T>
std::istream& operator>>(std::istream& istrm, Vector<T>& v)
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
inline Vector<T> operator+(const Matrix<T>& a, const Matrix<T>& b)
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

template <typename T>
std::ostream& operator<<(std::ostream& ostrm, const Matrix<T>& m)
{
    ostrm << m.rows() << " x " << m.cols() << '\n' << "{ ";
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            ostrm << std::setw(9) << m(i, j) << " ";
        }
        if (i != (m.rows() - 1)) {
            ostrm << "\n ";
        }
    }
    ostrm << '}';
    return ostrm;
}

template <typename T>
std::istream& operator>>(std::istream& istrm, Matrix<T>& m)
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

//------------------------------------------------------------------------------
// Matrix-matrix product:

template <typename T>
void matrix_matrix_product(const Matrix<T>& a,
                           const Matrix<T>& b,
                           Matrix<T>& res)
{
    constexpr std::size_t n = a.extent(0);
    constexpr std::size_t m = a.extent(1);
    constexpr std::size_t p = b.extent(1);
    assert(m == b.extent(0));

    res.resize(n, p);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            res(i, j) = T{0};
            for (std::size_t k = 0; k < m; ++k) {
                res(i, j) += a(i, k) * b(k, j);
            }
        }
    }
}

} // namespace Scilib
