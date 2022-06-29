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
inline void matrix_product(
    stdex::mdspan<T_a, stdex::extents<std::size_t, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    stdex::mdspan<T_b, stdex::extents<std::size_t, nrows_b, ncols_b>, Layout_b, Accessor_b> b,
    stdex::mdspan<T_c, stdex::extents<std::size_t, nrows_c, ncols_c>, Layout_c, Accessor_c> c)
{
    static_assert(a.static_extent(1) == b.static_extent(0));

    using size_type = std::size_t;

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
inline void matrix_product(Sci::Matrix_view<double, Layout> a,
                           Sci::Matrix_view<double, Layout> b,
                           Sci::Matrix_view<double, Layout> c)
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
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data(), lda, b.data(),
                ldb, beta, c.data(), ldc);
}

template <class Layout>
inline void matrix_product(Sci::Matrix_view<const double, Layout> a,
                           Sci::Matrix_view<const double, Layout> b,
                           Sci::Matrix_view<double, Layout> c)
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
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data(), lda, b.data(),
                ldb, beta, c.data(), ldc);
}

#ifdef USE_MKL
template <class Layout>
inline void matrix_product(Sci::Matrix_view<std::complex<double>, Layout> a,
                           Sci::Matrix_view<std::complex<double>, Layout> b,
                           Sci::Matrix_view<std::complex<double>, Layout> c)
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
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data(), lda, b.data(),
                ldb, &beta, c.data(), ldc);
}

template <class Layout>
inline void matrix_product(Sci::Matrix_view<const std::complex<double>, Layout> a,
                           Sci::Matrix_view<const std::complex<double>, Layout> b,
                           Sci::Matrix_view<std::complex<double>, Layout> c)
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
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data(), lda, b.data(),
                ldb, &beta, c.data(), ldc);
}
#endif

template <class T, class Layout, class Allocator>
inline Sci::Matrix<T, Layout, Allocator> matrix_product(const Sci::Matrix<T, Layout, Allocator>& a,
                                                        const Sci::Matrix<T, Layout, Allocator>& b)
{
    using size_type = std::size_t;

    const size_type n = a.extent(0);
    const size_type p = b.extent(1);

    Sci::Matrix<T, Layout, Allocator> res(n, p);
    matrix_product(a.view(), b.view(), res.view());
    return res;
}

template <class T, class Layout, class Allocator>
inline void matrix_product(const Sci::Matrix<T, Layout, Allocator>& a,
                           const Sci::Matrix<T, Layout, Allocator>& b,
                           Sci::Matrix<T, Layout, Allocator>& c)
{
    matrix_product(a.view(), b.view(), c.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
