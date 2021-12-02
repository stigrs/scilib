// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <cassert>

double Scilib::Linalg::det(Matrix_view<double> a)
{
    assert(a.extent(0) == a.extent(1));

    double ddet = 0.0;
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(0));

    if (n == 1) {
        ddet = a(0, 0);
    }
    else if (n == 2) {
        ddet = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    }
    else { // use LU decomposition
        Scilib::Matrix<double> tmp(a);
        Scilib::Vector<BLAS_INT> ipiv(n);

        lu(tmp.view(), ipiv.view());

        BLAS_INT permut = 0;
        for (BLAS_INT i = 1; i <= n; ++i) {
            if (i != ipiv(i - 1)) { // Fortran uses base 1
                permut++;
            }
        }
        ddet = Scilib::Linalg::prod(Scilib::diag(tmp.view()));
        ddet *= std::pow(-1.0, static_cast<double>(permut));
    }
    return ddet;
}
