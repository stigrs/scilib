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
void matrix_matrix_product(const Matrix_view<T>& a,
                           const Matrix_view<T>& b,
                           Matrix_view<T>& res)
{
    const std::size_t n = a.extent(0);
    const std::size_t m = a.extent(1);
    const std::size_t p = b.extent(1);
    assert(m == b.extent(0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            res(i, j) = T{0};
            for (std::size_t k = 0; k < m; ++k) {
                res(i, j) += a(i, k) * b(k, j);
            }
        }
    }
}

inline void matrix_matrix_product(const Matrix_view<double>& a,
                                  const Matrix_view<double>& b,
                                  Matrix_view<double>& res)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const Index m = narrow_cast<Index>(a.extent(0));
    const Index n = narrow_cast<Index>(b.extent(1));
    const Index k = narrow_cast<Index>(a.extent(1));

    const Index lda = k;
    const Index ldb = n;
    const Index ldc = n;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                a.data(), lda, b.data(), ldb, beta, res.data(), ldc);
}

#ifdef USE_MKL
inline void matrix_matrix_product(const Matrix_view<std::complex<double>>& a,
                                  const Matrix_view<std::complex<double>>& b,
                                  Matrix_view<std::complex<double>>& res)
{
    constexpr std::complex<double> alpha = {1.0, 0.0};
    constexpr std::complex<double> beta = {0.0, 0.0};

    const Index m = narrow_cast<Index>(a.extent(0));
    const Index n = narrow_cast<Index>(b.extent(1));
    const Index k = narrow_cast<Index>(a.extent(1));

    const Index lda = k;
    const Index ldb = n;
    const Index ldc = n;

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                a.data(), lda, b.data(), ldb, &beta, res.data(), ldc);
}
#endif

template <typename T>
inline Matrix<T> matrix_matrix_product(const Matrix_view<T>& a,
                                       const Matrix_view<T>& b)
{
    const std::size_t n = a.extent(0);
    const std::size_t m = a.extent(1);
    const std::size_t p = b.extent(1);
    assert(m == b.extent(0));

    Matrix<T> res(n, p);
    matrix_matrix_product(a, b, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib
