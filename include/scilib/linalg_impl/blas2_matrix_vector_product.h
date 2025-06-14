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

namespace Mdspan = std::experimental;

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
    Mdspan::mdspan<T_a, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    Mdspan::mdspan<T_x, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    Mdspan::mdspan<T_y, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
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
    Mdspan::mdspan<double, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
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

    if constexpr (std::is_same_v<Layout_a, Mdspan::layout_left>) {
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
    Mdspan::mdspan<const double, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a>
        a,
    Mdspan::mdspan<const double, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
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

    if constexpr (std::is_same_v<Layout_a, Mdspan::layout_left>) {
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
    Mdspan::mdspan<std::complex<double>,
                  Mdspan::extents<IndexType_a, nrows_a, ncols_a>,
                  Layout_a,
                  Accessor_a> a,
    Mdspan::mdspan<std::complex<double>, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
    Mdspan::mdspan<std::complex<double>, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
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

    if constexpr (std::is_same_v<Layout_a, Mdspan::layout_left>) {
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
    Mdspan::mdspan<const std::complex<double>,
                  Mdspan::extents<IndexType_a, nrows_a, ncols_a>,
                  Layout_a,
                  Accessor_a> a,
    Mdspan::
        mdspan<const std::complex<double>, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Accessor_x>
            x,
    Mdspan::mdspan<std::complex<double>, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y)
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

    if constexpr (std::is_same_v<Layout_a, Mdspan::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
    }
    cblas_zgemv(matrix_layout, CblasNoTrans, m, n, &alpha, a.data_handle(), lda, x.data_handle(),
                incx, &beta, y.data_handle(), incy);
}

template <class T_a,
          class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Container_a,
          class IndexType_x,
          std::size_t ext_x,
          class T_x,
          class Layout_x,
          class Container_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y>
    requires(!std::is_const_v<T_y> && std::is_integral_v<IndexType_a> &&
             std::is_integral_v<IndexType_x> && std::is_integral_v<IndexType_y>)
inline void matrix_vector_product(
    const Sci::MDArray<T_a, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Container_a>&
        a,
    const Sci::MDArray<T_x, Mdspan::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
    Sci::MDArray<T_y, Mdspan::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y)
{
    Expects(y.size() == gsl::narrow_cast<std::size_t>(a.extent(0)));
    matrix_vector_product(a.to_mdspan(), x.to_mdspan(), y.to_mdspan());
}

template <class T, class Layout>
inline Sci::Vector<T, Layout> matrix_vector_product(const Sci::Matrix<T, Layout>& a,
                                                    const Sci::Vector<T, Layout>& x)
{
    Sci::Vector<T, Layout> res(a.extent(0));
    matrix_vector_product(a, x, res);
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
