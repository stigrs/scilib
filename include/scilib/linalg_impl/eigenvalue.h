// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EIGENVALUE_H
#define SCILIB_LINALG_EIGENVALUE_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <exception>
#include <cassert>
#include <complex>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Compute eigenvalues and eigenvectors of a real symmetric matrix.
template <class Layout>
inline void eigs(Sci::Matrix_view<double, Layout> a,
                 Sci::Vector_view<double, Layout> w,
                 double abstol = -1.0 /* use default value */)
{
    static_assert(a.is_contiguous());
    static_assert(w.is_contiguous());

    assert(a.extent(0) == a.extent(1));
    assert(w.extent(0) == a.extent(0));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT nselect = n;
    const BLAS_INT lda = n;
    const BLAS_INT ldz = nselect;

    BLAS_INT il = 1;
    BLAS_INT iu = n;
    BLAS_INT m;
    BLAS_INT info;

    double vl = 0.0;
    double vu = 0.0;

    Sci::Vector<BLAS_INT, Layout> isuppz(2 * n);
    Sci::Matrix<double, Layout> z(ldz, n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }

    info = LAPACKE_dsyevr(matrix_layout, 'V', 'A', 'U', n, a.data(), lda, vl,
                          vu, il, iu, abstol, &m, w.data(), z.data(), ldz,
                          isuppz.data());
    if (info != 0) {
        throw std::runtime_error("dsyevr failed");
    }
    Sci::copy(z.view(), a);
}

template <class Layout>
inline void eigs(Sci::Matrix<double, Layout>& a,
                 Sci::Vector<double, Layout>& w,
                 double abstol = -1.0 /* use default value */)
{
    eigs(a.view(), w.view(), abstol);
}

// Compute eigenvalues and eigenvectors of a real non-symmetric matrix.
template <class Layout>
void eig(Sci::Matrix_view<double, Layout> a,
         Sci::Matrix_view<std::complex<double>, Layout> evec,
         Sci::Vector_view<std::complex<double>, Layout> eval)
{
    using namespace Sci;

    static_assert(a.is_contiguous());
    static_assert(evec.is_contiguous());
    static_assert(eval.is_contiguous());

    assert(a.extent(0) == a.extent(1));
    assert(a.extent(0) == eval.extent(0));
    assert(a.extent(0) == evec.extent(0));
    assert(a.extent(1) == evec.extent(1));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    Sci::Vector<double, Layout> wr(n);
    Sci::Vector<double, Layout> wi(n);
    Sci::Matrix<double, Layout> vr(n, n);
    Sci::Matrix<double, Layout> vl(n, n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }
    BLAS_INT info =
        LAPACKE_dgeev(matrix_layout, 'N', 'V', n, a.data(), n, wr.data(),
                      wi.data(), vl.data(), n, vr.data(), n);
    if (info != 0) {
        throw std::runtime_error("dgeev failed");
    }
    for (BLAS_INT i = 0; i < n; ++i) {
        std::complex<double> wii(wr(i), wi(i));
        eval(i) = wii;
        BLAS_INT j = 0;
        while (j < n) {
            if (wi(j) == 0.0) {
                evec(i, j) = std::complex<double>{vr(i, j), 0.0};
                ++j;
            }
            else {
                evec(i, j) = std::complex<double>{vr(i, j), vr(i, j + 1)};
                evec(i, j + 1) = std::complex<double>{vr(i, j), -vr(i, j + 1)};
                j += 2;
            }
        }
    }
}

template <class Layout>
void eig(Sci::Matrix<double, Layout>& a,
         Sci::Matrix<std::complex<double>, Layout>& evec,
         Sci::Vector<std::complex<double>, Layout>& eval)
{
    eig(a.view(), evec.view(), eval.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_EIGENVALUE_H
