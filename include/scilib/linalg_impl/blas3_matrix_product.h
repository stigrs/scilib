// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
#define SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H

#include "lapack_types.h"
#include <complex>
#include <experimental/linalg>
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
          class T_b,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Layout_b,
          class Accessor_b,
          class T_c,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Layout_c,
          class Accessor_c>
    requires(!std::is_const_v<T_c> && std::is_integral_v<IndexType_a> &&
             std::is_integral_v<IndexType_b> && std::is_integral_v<IndexType_c>)
inline void matrix_product(
    Mdspan::mdspan<T_a, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    Mdspan::mdspan<T_b, Mdspan::extents<IndexType_b, nrows_b, ncols_b>, Layout_b, Accessor_b> b,
    Mdspan::mdspan<T_c, Mdspan::extents<IndexType_c, nrows_c, ncols_c>, Layout_c, Accessor_c> c)
{
    std::experimental::linalg::matrix_product(a, b, c);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>&&
                 std::is_integral_v<IndexType_c>)
inline void matrix_product(
    Mdspan::mdspan<double, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_b, nrows_b, ncols_b>, Layout, Accessor_b> b,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_c, nrows_c, ncols_c>, Layout, Accessor_c> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, Mdspan::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data_handle(), lda,
                b.data_handle(), ldb, beta, c.data_handle(), ldc);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>&&
                 std::is_integral_v<IndexType_c>)
inline void matrix_product(
    Mdspan::mdspan<const double, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a>
        a,
    Mdspan::mdspan<const double, Mdspan::extents<IndexType_b, nrows_b, ncols_b>, Layout, Accessor_b>
        b,
    Mdspan::mdspan<double, Mdspan::extents<IndexType_c, nrows_c, ncols_c>, Layout, Accessor_c> c)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, Mdspan::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_dgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data_handle(), lda,
                b.data_handle(), ldb, beta, c.data_handle(), ldc);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>&&
                 std::is_integral_v<IndexType_c>)
inline void matrix_product(Mdspan::mdspan<std::complex<double>,
                                         Mdspan::extents<IndexType_a, nrows_a, ncols_a>,
                                         Layout,
                                         Accessor_a> a,
                           Mdspan::mdspan<std::complex<double>,
                                         Mdspan::extents<IndexType_b, nrows_b, ncols_b>,
                                         Layout,
                                         Accessor_b> b,
                           Mdspan::mdspan<std::complex<double>,
                                         Mdspan::extents<IndexType_c, nrows_c, ncols_c>,
                                         Layout,
                                         Accessor_c> c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, Mdspan::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data(), lda, b.data(),
                ldb, &beta, c.data(), ldc);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Accessor_c>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>&&
                 std::is_integral_v<IndexType_c>)
inline void matrix_product(Mdspan::mdspan<const std::complex<double>,
                                         Mdspan::extents<IndexType_a, nrows_a, ncols_a>,
                                         Layout,
                                         Accessor_a> a,
                           Mdspan::mdspan<const std::complex<double>,
                                         Mdspan::extents<IndexType_b, nrows_b, ncols_b>,
                                         Layout,
                                         Accessor_b> b,
                           Mdspan::mdspan<std::complex<double>,
                                         Mdspan::extents<IndexType_c, nrows_c, ncols_c>,
                                         Layout,
                                         Accessor_c> c)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT k = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = CblasRowMajor;
    BLAS_INT lda = k;
    BLAS_INT ldb = n;
    BLAS_INT ldc = n;

    if constexpr (std::is_same_v<Layout, Mdspan::layout_left>) {
        matrix_layout = CblasColMajor;
        lda = m;
        ldb = k;
        ldc = m;
    }
    cblas_zgemm(matrix_layout, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a.data_handle(), lda,
                b.data_handle(), ldb, &beta, c.data_handle(), ldc);
}

template <class T_a,
          class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout_a,
          class Container_a,
          class T_b,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Layout_b,
          class Container_b,
          class T_c,
          class IndexType_c,
          std::size_t nrows_c,
          std::size_t ncols_c,
          class Layout_c,
          class Container_c>
    requires(!std::is_const_v<T_c> && std::is_integral_v<IndexType_a> &&
             std::is_integral_v<IndexType_b> && std::is_integral_v<IndexType_c>)
inline void matrix_product(
    const Sci::MDArray<T_a, Mdspan::extents<IndexType_a, nrows_a, ncols_a>, Layout_a, Container_a>&
        a,
    const Sci::MDArray<T_b, Mdspan::extents<IndexType_b, nrows_b, ncols_b>, Layout_b, Container_b>&
        b,
    Sci::MDArray<T_c, Mdspan::extents<IndexType_c, nrows_c, ncols_c>, Layout_c, Container_c>& c)
{
    matrix_product(a.to_mdspan(), b.to_mdspan(), c.to_mdspan());
}

template <class T, class Layout>
inline Sci::Matrix<T, Layout> matrix_product(const Sci::Matrix<T, Layout>& a,
                                             const Sci::Matrix<T, Layout>& b)
{
    using index_type = typename Sci::Matrix<T, Layout>::index_type;

    const index_type n = a.extent(0);
    const index_type p = b.extent(1);

    Sci::Matrix<T, Layout> res(n, p);
    matrix_product(a.to_mdspan(), b.to_mdspan(), res.to_mdspan());
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS3_MATRIX_PRODUCT_H
