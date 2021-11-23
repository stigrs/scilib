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

#include <scilib/mdarray_impl/matrix.h>
#include <scilib/traits.h>
#include <cassert>
#include <complex>

namespace Scilib {
namespace Linalg {

template <typename T>
void matrix_vector_product(const Matrix_view<T>& a,
                           const Vector_view<T>& x,
                           Vector_view<T>& y)
{
    assert(x.size() == a.extent(1));
    for (std::size_t i = 0; i < a.extent(0); ++i) {
        y(i) = T{0};
        for (std::size_t j = 0; j < a.extent(1); ++j) {
            y(i) += a(i, j) * x(j);
        }
    }
}

inline void matrix_vector_product(const Matrix_view<double>& a,
                                  const Vector_view<double>& x,
                                  Vector_view<double>& y)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    assert(x.size() == a.extent(1));

    const Index m = narrow_cast<Index>(a.extent(0));
    const Index n = narrow_cast<Index>(a.extent(1));

    const Index lda = n;
    const Index incx = narrow_cast<Index>(x.stride(0));
    const Index incy = narrow_cast<Index>(y.stride(0));

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a.data(), lda,
                x.data(), incx, beta, y.data(), incy);
}

#ifdef USE_MKL
inline void matrix_vector_product(const Matrix_view<std::complex<double>>& a,
                                  const Vector_view<std::complex<double>>& x,
                                  Vector_view<std::complex<double>>& y)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    assert(x.size() == a.extent(1));

    const Index m = narrow_cast<Index>(a.extent(0));
    const Index n = narrow_cast<Index>(a.extent(1));

    const Index lda = n;
    const Index incx = narrow_cast<Index>(x.stride(0));
    const Index incy = narrow_cast<Index>(y.stride(0));

    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, &alpha, a.data(), lda,
                x.data(), incx, &beta, y.data(), incy);
}
#endif

template <typename T>
inline Vector<T> matrix_vector_product(const Matrix_view<T>& a,
                                       const Vector_view<T>& x)
{
    Vector<T> res(narrow_cast<Index>(a.extent(0)));
    matrix_vector_product(a, x, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib
