// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_DET_H
#define SCILIB_LINALG_DET_H

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/auxiliary.h>
#include <scilib/linalg_impl/matrix_decomposition.h>
#include <cassert>
#include <type_traits>

namespace Scilib {
namespace Linalg {

// Determinant of square matrix.
// clang-format off
template <class T, class Layout>
    requires std::is_same_v<std::remove_cv_t<T>, double>
auto det(Scilib::Matrix_view<T, Layout> a)
// clang-format on
{
    static_assert(a.is_contiguous());
    assert(a.extent(0) == a.extent(1));

    using value_type = std::remove_cv_t<T>;

    value_type ddet = 0.0;
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(0));

    if (n == 1) {
        ddet = a(0, 0);
    }
    else if (n == 2) {
        ddet = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    }
    else { // use LU decomposition
        Scilib::Matrix<value_type, Layout> tmp(a);
        Scilib::Vector<BLAS_INT, Layout> ipiv(n);

        Scilib::Linalg::lu(tmp.view(), ipiv.view());

        BLAS_INT permut = 0;
        for (BLAS_INT i = 1; i <= n; ++i) {
            if (i != ipiv(i - 1)) { // Fortran uses base 1
                permut++;
            }
        }
        ddet = Scilib::Linalg::prod(Scilib::diag(tmp.view()));
        ddet *= std::pow(-1.0, static_cast<value_type>(permut));
    }
    return ddet;
}

// clang-format off
template <class T, class Layout>
    requires std::is_same_v<std::remove_cv_t<T>, double>
inline T det(const Scilib::Matrix<T, Layout>& a)
// clang-format on
{
    return det(a.view());
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_DET_H
