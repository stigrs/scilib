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

#include <scilib/traits.h>
#include <scilib/mdarray_impl/type_aliases.h>
#include <experimental/mdspan>
#include <complex>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type nrows_a,
          stdex::extents<>::size_type ncols_a,
          class Layout_a,
          class Accessor_a,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void matrix_vector_product(
    stdex::mdspan<T, stdex::extents<nrows_a, ncols_a>, Layout_a, Accessor_a> a,
    stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == a.static_extent(1));
    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < a.extent(0); ++i) {
        y(i) = T{0};
        for (size_type j = 0; j < a.extent(1); ++j) {
            y(i) += a(i, j) * x(j);
        }
    }
}

inline void matrix_vector_product(Matrix_view<double> a,
                                  Vector_view<double> x,
                                  Vector_view<double> y)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    static_assert(x.static_extent(0) == a.static_extent(1));

    const BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));

    const BLAS_INT lda = n;
    const BLAS_INT incx = narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = narrow_cast<BLAS_INT>(y.stride(0));

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a.data(), lda,
                x.data(), incx, beta, y.data(), incy);
}

#ifdef USE_MKL
inline void matrix_vector_product(Matrix_view<std::complex<double>> a,
                                  Vector_view<std::complex<double>> x,
                                  Vector_view<std::complex<double>> y)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    assert(x.size() == a.extent(1));

    const BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));

    const BLAS_INT lda = n;
    const BLAS_INT incx = narrow_cast<BLAS_INT>(x.stride(0));
    const BLAS_INT incy = narrow_cast<BLAS_INT>(y.stride(0));

    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, &alpha, a.data(), lda,
                x.data(), incx, &beta, y.data(), incy);
}
#endif

template <typename T>
inline Vector<T> matrix_vector_product(Matrix_view<T> a, Vector_view<T> x)
{
    Vector<T> res(narrow_cast<BLAS_INT>(a.extent(0)));
    matrix_vector_product(a, x, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS2_MATRIX_VECTOR_PRODUCT_H
