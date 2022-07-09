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

#include "lapack_types.h"
#include <complex>
#include <type_traits>

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class T_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Layout_b,
          class Accessor_b,
          class T_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Layout_c,
          class Accessor_c>
    requires(!std::is_const_v<T_c>)
inline void
matrix_product(stdex::mdspan<T_a, stdex::extents<index, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
               stdex::mdspan<T_b, stdex::extents<index, nrows_b, ncols_b>, Layout_b, Accessor_b> b,
               stdex::mdspan<T_c, stdex::extents<index, nrows_c, ncols_c>, Layout_c, Accessor_c> c)
{
    std::experimental::linalg::matrix_product(a, b, c);
}

template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
inline void
matrix_product(stdex::mdspan<double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
               stdex::mdspan<double, stdex::extents<index, nrows_b, ncols_b>, Layout, Accessor_b> b,
               stdex::mdspan<double, stdex::extents<index, nrows_c, ncols_c>, Layout, Accessor_c> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow<BLAS_INT>(a.extent(1));

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
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data_handle(), lda,
                b.data_handle(), ldb, beta, c.data_handle(), ldc);
}

template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
inline void matrix_product(
    stdex::mdspan<const double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
    stdex::mdspan<const double, stdex::extents<index, nrows_b, ncols_b>, Layout, Accessor_b> b,
    stdex::mdspan<double, stdex::extents<index, nrows_c, ncols_c>, Layout, Accessor_c> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow<BLAS_INT>(a.extent(1));

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
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data_handle(), lda,
                b.data_handle(), ldb, beta, c.data_handle(), ldc);
}

#ifdef USE_MKL
template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
inline void matrix_product(
    stdex::mdspan<std::complex<double>, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a>
        a,
    stdex::mdspan<std::complex<double>, stdex::extents<index, nrows_b, ncols_b>, Layout, Accessor_b>
        b,
    stdex::mdspan<std::complex<double>, stdex::extents<index, nrows_c, ncols_c>, Layout, Accessor_c>
        c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow<BLAS_INT>(a.extent(1));

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
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data(), lda, b.data(),
                ldb, &beta, c.data(), ldc);
}

template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
inline void matrix_product(
    stdex::mdspan<const std::complex<double>,
                  stdex::extents<index, nrows_a, ncols_a>,
                  Layout,
                  Accessor_a> a,
    stdex::mdspan<const std::complex<double>,
                  stdex::extents<index, nrows_b, ncols_b>,
                  Layout,
                  Accessor_b> b,
    stdex::mdspan<std::complex<double>, stdex::extents<index, nrows_c, ncols_c>, Layout, Accessor_c>
        c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow<BLAS_INT>(a.extent(1));

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
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data_handle(), lda,
                b.data_handle(), ldb, &beta, c.data_handle(), ldc);
}
#endif

template <class T, class Layout, class Container>
inline Sci::Matrix<T, Layout, Container> matrix_product(const Sci::Matrix<T, Layout, Container>& a,
                                                        const Sci::Matrix<T, Layout, Container>& b)
{
    using index_type = index;

    const index_type n = a.extent(0);
    const index_type p = b.extent(1);

    Sci::Matrix<T, Layout, Container> res(n, p);
    matrix_product(a.view(), b.view(), res.view());
    return res;
}

template <class T, class Layout, class Container>
inline void matrix_product(const Sci::Matrix<T, Layout, Container>& a,
                           const Sci::Matrix<T, Layout, Container>& b,
                           Sci::Matrix<T, Layout, Container>& c)
{
    matrix_product(a.view(), b.view(), c.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
