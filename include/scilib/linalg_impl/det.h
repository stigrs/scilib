// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_DET_H
#define SCILIB_LINALG_DET_H

#include "auxiliary.h"
#include "matrix_decomposition.h"
#include <cassert>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Determinant of square matrix.
template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
    requires(std::is_same_v<std::remove_cv_t<T>, double>)
auto det(stdex::mdspan<T, stdex::extents<index, nrows, ncols>, Layout, Accessor> a)
{
    Expects(a.extent(0) == a.extent(1));

    using value_type = std::remove_cv_t<T>;

    value_type ddet = 0.0;
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(0));

    if (n == 1) {
        ddet = a(0, 0);
    }
    else if (n == 2) {
        ddet = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    }
    else { // use LU decomposition
        Sci::Matrix<value_type, Layout> tmp(a);
        Sci::Vector<BLAS_INT, Layout> ipiv(n);

        Sci::Linalg::lu(tmp.view(), ipiv.view());

        BLAS_INT permut = 0;
        for (BLAS_INT i = 1; i <= n; ++i) {
            if (i != ipiv(i - 1)) { // Fortran uses base 1
                permut++;
            }
        }
        ddet = Sci::Linalg::prod(Sci::diag(tmp.view()));
        ddet *= std::pow(-1.0, gsl::narrow_cast<value_type>(permut));
    }
    return ddet;
}

template <class T, class Layout, class Container>
    requires(std::is_same_v<std::remove_cv_t<T>, double>)
inline T det(const Sci::Matrix<T, Layout, Container>& a) { return det(a.view()); }

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_DET_H
