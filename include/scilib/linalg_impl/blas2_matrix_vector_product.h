// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
#define SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <experimental/mdspan>
#include <complex>
#include <type_traits>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_a,
          stdex::extents<>::size_type nrows_a,
          stdex::extents<>::size_type ncols_a,
          class Layout_a,
          class Accessor_a,
          class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
// clang-format off
    requires !std::is_const_v<T_y>
inline void matrix_vector_product(
    // clang-format on
    stdex::mdspan<T_a, stdex::extents<nrows_a, ncols_a>, Layout_a, Accessor_a>
        a,
    stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < a.extent(0); ++i) {
        y(i) = T_y{0};
        for (size_type j = 0; j < a.extent(1); ++j) {
            y(i) += a(i, j) * x(j);
        }
    }
}

inline void matrix_vector_product(Scilib::Matrix_view<double> a,
                                  Scilib::Vector_view<double> x,
                                  Scilib::Vector_view<double> y)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    static_assert(x.static_extent(0) == a.static_extent(1));

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    const BLAS_INT lda = n;
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a.data(), lda,
                x.data(), incx, beta, y.data(), incy);
}

#ifdef USE_MKL
// Does not work with OpenBLAS version 0.2.14.1
inline void matrix_vector_product(Scilib::Matrix_view<std::complex<double>> a,
                                  Scilib::Vector_view<std::complex<double>> x,
                                  Scilib::Vector_view<std::complex<double>> y)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    static_assert(x.static_extent(0) == a.static_extent(1));

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    const BLAS_INT lda = n;
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, &alpha, a.data(), lda,
                x.data(), incx, &beta, y.data(), incy);
}
#endif

template <class T>
inline Scilib::Vector<T> matrix_vector_product(Scilib::Matrix_view<T> a,
                                               Scilib::Vector_view<T> x)
{
    Scilib::Vector<T> res(static_cast<BLAS_INT>(a.extent(0)));
    matrix_vector_product(a, x, res.view());
    return res;
}

template <class T>
inline Scilib::Vector<T> matrix_vector_product(Scilib::Matrix_view<const T> a,
                                               Scilib::Vector_view<const T> x)
{
    Scilib::Vector<T> res(static_cast<BLAS_INT>(a.extent(0)));
    matrix_vector_product(a, x, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
