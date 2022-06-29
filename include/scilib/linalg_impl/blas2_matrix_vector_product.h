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

#include "lapack_types.h"
#include <cassert>
#include <complex>
#include <type_traits>
#include <iostream>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class T_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y>)
inline void matrix_vector_product(
    stdex::mdspan<T_a,
                  stdex::extents<std::size_t, nrows_a, ncols_a>,
                  Layout_a,
                  Accessor_a> a,
    stdex::mdspan<T_x, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x>
        x,
    stdex::mdspan<T_y, stdex::extents<std::size_t, ext_y>, Layout_y, Accessor_y>
        y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));
    using size_type = std::size_t;

    for (size_type i = 0; i < a.extent(0); ++i) {
        y(i) = T_y{0};
        for (size_type j = 0; j < a.extent(1); ++j) {
            y(i) += a(i, j) * x(j);
        }
    }
}

template <class Layout>
inline void matrix_vector_product(Sci::Matrix_view<double, Layout> a,
                                  Sci::Vector_view<double, Layout> x,
                                  Sci::Vector_view<double, Layout> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_dgemv(matrix_layout, CblasNoTrans, m, n, alpha, a.data(), lda,
                x.data(), incx, beta, y.data(), incy);
}

template <class Layout>
inline void matrix_vector_product(Sci::Matrix_view<const double, Layout> a,
                                  Sci::Vector_view<const double, Layout> x,
                                  Sci::Vector_view<double, Layout> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_dgemv(matrix_layout, CblasNoTrans, m, n, alpha, a.data(), lda,
                x.data(), incx, beta, y.data(), incy);
}

#ifdef USE_MKL
// Does not work with OpenBLAS version 0.2.14.1
template <class Layout>
inline void
matrix_vector_product(Sci::Matrix_view<std::complex<double>, Layout> a,
                      Sci::Vector_view<std::complex<double>, Layout> x,
                      Sci::Vector_view<std::complex<double>, Layout> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_zgemv(matrix_layout, CblasNoTrans, m, n, &alpha, a.data(), lda,
                x.data(), incx, &beta, y.data(), incy);
}

template <class Layout>
inline void
matrix_vector_product(Sci::Matrix_view<const std::complex<double>, Layout> a,
                      Sci::Vector_view<const std::complex<double>, Layout> x,
                      Sci::Vector_view<std::complex<double>, Layout> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = static_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_zgemv(matrix_layout, CblasNoTrans, m, n, &alpha, a.data(), lda,
                x.data(), incx, &beta, y.data(), incy);
}
#endif

template <class T, class Layout, class Allocator>
inline Sci::Vector<T, Layout, Allocator>
matrix_vector_product(const Sci::Matrix<T, Layout, Allocator>& a,
                      const Sci::Vector<T, Layout, Allocator>& x)
{
    Sci::Vector<T, Layout, Allocator> res(a.extent(0));
    matrix_vector_product(a.view(), x.view(), res.view());
    return res;
}

template <class T, class Layout, class Allocator>
inline void matrix_vector_product(const Sci::Matrix<T, Layout, Allocator>& a,
                                  const Sci::Vector<T, Layout, Allocator>& x,
                                  Sci::Vector<T, Layout, Allocator>& res)
{
    assert(res.size() == a.extent(0));
    matrix_vector_product(a.view(), x.view(), res.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
