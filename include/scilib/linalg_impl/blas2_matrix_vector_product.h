// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
#define SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H

#include "lapack_types.h"
#include <cassert>
#include <complex>
#include <experimental/linalg>
#include <gsl/gsl>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_a,
          class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y> && std::is_integral_v<IndexType_a> &&
             std::is_integral_v<IndexType_x> && std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    stdex::mdspan<T_a, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    stdex::mdspan<T_x, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    std::experimental::linalg::matrix_vector_product(a, x, y);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_x>&&
                 std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    stdex::mdspan<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    stdex::mdspan<double, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<double, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = gsl::narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = gsl::narrow_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout_a, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_dgemv(matrix_layout, CblasNoTrans, m, n, alpha, a.data_handle(), lda, x.data_handle(),
                incx, beta, y.data_handle(), incy);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_x>&&
                 std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    stdex::mdspan<const double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a>
        a,
    stdex::mdspan<const double, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<double, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = gsl::narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = gsl::narrow_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout_a, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_dgemv(matrix_layout, CblasNoTrans, m, n, alpha, a.data_handle(), lda, x.data_handle(),
                incx, beta, y.data_handle(), incy);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_x>&&
                 std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    stdex::mdspan<std::complex<double>,
                  stdex::extents<IndexType_a, nrows_a, ncols_a>,
                  Layout_a,
                  Accessor_a> a,
    stdex::mdspan<std::complex<double>, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<std::complex<double>, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = gsl::narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = gsl::narrow_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout_a, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_zgemv(matrix_layout, CblasNoTrans, m, n, &alpha, a.data_handle(), lda, x.data_handle(),
                incx, &beta, y.data_handle(), incy);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Accessor_a,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_x>&&
                 std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    stdex::mdspan<const std::complex<double>,
                  stdex::extents<IndexType_a, nrows_a, ncols_a>,
                  Layout_a,
                  Accessor_a> a,
    stdex::
        mdspan<const std::complex<double>, stdex::extents<IndexType_x, ext_x>, Layout_x, Accessor_x>
            x,
    stdex::mdspan<std::complex<double>, stdex::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));

    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT incx = gsl::narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = gsl::narrow_cast<BLAS_INT>(y.stride(0));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout_a, stdex::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_zgemv(matrix_layout, CblasNoTrans, m, n, &alpha, a.data_handle(), lda, x.data_handle(),
                incx, &beta, y.data_handle(), incy);
}

template <class T,
          class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Container_a,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Container_x,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y>
inline void matrix_vector_product(
    const Sci::MDArray<T, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Container_a>& a,
    const Sci::MDArray<T, stdex::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
    Sci::MDArray<T, stdex::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y)
{
    Expects(y.size() == gsl::narrow_cast<std::size_t>(a.extent(0)));
    matrix_vector_product(a.view(), x.view(), y.view());
}

template <class T, class Layout, class Container>
inline Sci::Vector<T, Layout, Container>
matrix_vector_product(const Sci::Matrix<T, Layout, Container>& a,
                      const Sci::Vector<T, Layout, Container>& x)
{
    Sci::Vector<T, Layout, Container> res(a.extent(0));
    matrix_vector_product(a, x, res);
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
