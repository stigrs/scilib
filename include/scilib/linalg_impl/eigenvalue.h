// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EIGENVALUE_H
#define SCILIB_LINALG_EIGENVALUE_H

#include "lapack_types.h"
#include <cassert>
#include <complex>
#include <exception>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Compute eigenvalues and eigenvectors of a real symmetric matrix.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_w,
          std::size_t ext_w,
          class Accessor_w>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_w>)
inline void
eigh(stdex::mdspan<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
     stdex::mdspan<double, stdex::extents<IndexType_w, ext_w>, Layout, Accessor_w> w,
     char uplo = 'U',
     double abstol = -1.0 /* use default value */)
{
    Expects(a.extent(0) == a.extent(1));
    Expects(w.extent(0) == a.extent(0));

    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(0));
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

    info = LAPACKE_dsyevr(matrix_layout, 'V', 'A', uplo, n, a.data_handle(), lda, vl, vu, il, iu,
                          abstol, &m, w.data_handle(), z.container_data(), ldz,
                          isuppz.container_data());
    if (info != 0) {
        throw std::runtime_error("dsyevr failed");
    }
    Sci::copy(z.to_mdspan(), a);
}

// Compute eigenvalues and eigenvectors of a complex Hermitian matrix.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_w,
          std::size_t ext_w,
          class Accessor_w>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_w>)
inline void eigh(stdex::mdspan<std::complex<double>,
                               stdex::extents<IndexType_a, nrows_a, ncols_a>,
                               Layout,
                               Accessor_a> a,
                 stdex::mdspan<double, stdex::extents<IndexType_w, ext_w>, Layout, Accessor_w> w,
                 char uplo = 'U',
                 double abstol = -1.0 /* use default value */)
{
    Expects(a.extent(0) == a.extent(1));
    Expects(w.extent(0) == a.extent(0));

    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;
    const BLAS_INT ldz = n;

    BLAS_INT il = 1;
    BLAS_INT iu = n;
    BLAS_INT m;
    BLAS_INT info;

    double vl = 0.0;
    double vu = 0.0;

    Sci::Vector<BLAS_INT, Layout> isuppz(2 * n);
    Sci::Matrix<std::complex<double>, Layout> z(ldz, n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }

    info = LAPACKE_zheevr(matrix_layout, 'V', 'A', uplo, n, a.data_handle(), lda, vl, vu, il, iu,
                          abstol, &m, w.data_handle(), z.container_data(), ldz,
                          isuppz.container_data());
    if (info != 0) {
        throw std::runtime_error("zheevr failed");
    }
    Sci::copy(z.to_mdspan(), a);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_w,
          std::size_t ext_w,
          class Container_w>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_w>)
inline void
eigh(Sci::MDArray<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Container_a>& a,
     Sci::MDArray<double, stdex::extents<IndexType_w, ext_w>, Layout, Container_w>& w,
     char uplo = 'U',
     double abstol = -1.0 /* use default value */)
{
    eigh(a.to_mdspan(), w.to_mdspan(), uplo, abstol);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_w,
          std::size_t ext_w,
          class Container_w>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_w>)
inline void eigh(Sci::MDArray<std::complex<double>,
                              stdex::extents<IndexType_a, nrows_a, ncols_a>,
                              Layout,
                              Container_a>& a,
                 Sci::MDArray<double, stdex::extents<IndexType_w, ext_w>, Layout, Container_w>& w,
                 char uplo = 'U',
                 double abstol = -1.0 /* use default value */)
{
    eigh(a.to_mdspan(), w.to_mdspan(), uplo, abstol);
}

// Compute eigenvalues and eigenvectors of a real non-symmetric matrix.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_evec,
          std::size_t nrows_evec,
          std::size_t ncols_evec,
          class Accessor_evec,
          class IndexType_eval,
          std::size_t ext_eval,
          class Accessor_eval>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_evec>&&
                 std::is_integral_v<IndexType_eval>)
void eig(stdex::mdspan<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
         stdex::mdspan<std::complex<double>,
                       stdex::extents<IndexType_evec, nrows_evec, ncols_evec>,
                       Layout,
                       Accessor_evec> evec,
         stdex::mdspan<std::complex<double>,
                       stdex::extents<IndexType_eval, ext_eval>,
                       Layout,
                       Accessor_eval> eval)
{
    using namespace Sci;

    Expects(a.extent(0) == a.extent(1));
    Expects(a.extent(0) == eval.extent(0));
    Expects(a.extent(0) == evec.extent(0));
    Expects(a.extent(1) == evec.extent(1));

    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    Sci::Vector<double, Layout> wr(n);
    Sci::Vector<double, Layout> wi(n);
    Sci::Matrix<double, Layout> vr(n, n);
    Sci::Matrix<double, Layout> vl(n, n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }
    BLAS_INT info =
        LAPACKE_dgeev(matrix_layout, 'N', 'V', n, a.data_handle(), n, wr.container_data(),
                      wi.container_data(), vl.container_data(), n, vr.container_data(), n);
    if (info != 0) {
        throw std::runtime_error("dgeev failed");
    }
    for (BLAS_INT i = 0; i < n; ++i) {
        std::complex<double> wii(wr[i], wi[i]);
        eval[i] = wii;
        BLAS_INT j = 0;
        while (j < n) {
            if (wi[j] == 0.0) {
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

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_evec,
          std::size_t nrows_evec,
          std::size_t ncols_evec,
          class Container_evec,
          class IndexType_eval,
          std::size_t ext_eval,
          class Container_eval>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_evec>&&
                 std::is_integral_v<IndexType_eval>)
void eig(
    Sci::MDArray<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Container_a>& a,
    Sci::MDArray<std::complex<double>,
                 stdex::extents<IndexType_evec, nrows_evec, ncols_evec>,
                 Layout,
                 Container_evec>& evec,
    Sci::MDArray<std::complex<double>,
                 stdex::extents<IndexType_eval, ext_eval>,
                 Layout,
                 Container_eval>& eval)
{
    eig(a.to_mdspan(), evec.to_mdspan(), eval.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_EIGENVALUE_H
