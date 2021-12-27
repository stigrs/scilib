// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
#define SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H

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
          class T_b,
          stdex::extents<>::size_type nrows_b,
          stdex::extents<>::size_type ncols_b,
          class Layout_b,
          class Accessor_b,
          class T_c,
          stdex::extents<>::size_type nrows_c,
          stdex::extents<>::size_type ncols_c,
          class Layout_c,
          class Accessor_c>
// clang-format off
    requires (!std::is_const_v<T_c>)
inline void matrix_product(
    // clang-format on
    stdex::mdspan<T_a, stdex::extents<nrows_a, ncols_a>, Layout_a, Accessor_a>
        a,
    stdex::mdspan<T_b, stdex::extents<nrows_b, ncols_b>, Layout_b, Accessor_b>
        b,
    stdex::mdspan<T_c, stdex::extents<nrows_c, ncols_c>, Layout_c, Accessor_c>
        c)
{
    static_assert(a.static_extent(1) == b.static_extent(0));

    using size_type = stdex::extents<>::size_type;

    const size_type n = a.extent(0);
    const size_type m = a.extent(1);
    const size_type p = b.extent(1);

    for (size_type i = 0; i < n; ++i) {
        for (size_type j = 0; j < p; ++j) {
            c(i, j) = T_c{0};
            for (size_type k = 0; k < m; ++k) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
}

template <class Layout>
inline void matrix_product(Scilib::Matrix_view<double, Layout> a,
                           Scilib::Matrix_view<double, Layout> b,
                           Scilib::Matrix_view<double, Layout> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = static_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
}

template <class Layout>
inline void matrix_product(Scilib::Matrix_view<const double, Layout> a,
                           Scilib::Matrix_view<const double, Layout> b,
                           Scilib::Matrix_view<double, Layout> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = static_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
}

#ifdef USE_MKL
template <class Layout>
inline void matrix_product(Scilib::Matrix_view<std::complex<double>, Layout> a,
                           Scilib::Matrix_view<std::complex<double>, Layout> b,
                           Scilib::Matrix_view<std::complex<double>, Layout> c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = static_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
}

template <class Layout>
inline void
matrix_product(Scilib::Matrix_view<const std::complex<double>, Layout> a,
               Scilib::Matrix_view<const std::complex<double>, Layout> b,
               Scilib::Matrix_view<std::complex<double>, Layout> c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = static_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
}
#endif

template <class T, class Layout>
inline Scilib::Matrix<T, Layout>
matrix_product(Scilib::Matrix_view<T, Layout> a,
               Scilib::Matrix_view<T, Layout> b)
{
    using size_type = stdex::extents<>::size_type;

    const size_type n = a.extent(0);
    const size_type p = b.extent(1);

    Scilib::Matrix<T, Layout> res(n, p);
    matrix_product(a, b, res.view());
    return res;
}

template <class T, class Layout>
inline Scilib::Matrix<T, Layout>
matrix_product(Scilib::Matrix_view<const T, Layout> a,
               Scilib::Matrix_view<const T, Layout> b)
{
    using size_type = stdex::extents<>::size_type;

    const size_type n = a.extent(0);
    const size_type p = b.extent(1);

    Scilib::Matrix<T, Layout> res(n, p);
    matrix_product(a, b, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
