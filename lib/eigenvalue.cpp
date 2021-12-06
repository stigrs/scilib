// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <exception>

void Scilib::Linalg::eig(Scilib::Matrix_view<double> a,
                         Scilib::Matrix_view<std::complex<double>> evec,
                         Scilib::Vector_view<std::complex<double>> eval)
{
    using namespace Scilib;

    static_assert(a.is_contiguous());
    static_assert(evec.is_contiguous());
    static_assert(eval.is_contiguous());

    assert(a.extent(0) == a.extent(1));
    assert(a.extent(0) == eval.extent(0));
    assert(a.extent(0) == evec.extent(0));
    assert(a.extent(1) == evec.extent(1));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    Scilib::Vector<double> wr(n);
    Scilib::Vector<double> wi(n);
    Scilib::Matrix<double> vr(n, n);
    Scilib::Matrix<double> vl(n, n);

    BLAS_INT info =
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, a.data(), n, wr.data(),
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
