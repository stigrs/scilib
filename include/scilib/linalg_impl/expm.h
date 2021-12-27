// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EXPM_H
#define SCILIB_LINALG_EXPM_H

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/auxiliary.h>
#include <scilib/linalg_impl/matrix_norm.h>
#include <scilib/linalg_impl/scaled.h>
#include <cassert>

namespace Scilib {
namespace Linalg {

template <class Layout>
Scilib::Matrix<double, Layout> expm(Scilib::Matrix_view<double, Layout> a)
{
    // Algorithm: Matlab expm1 (demo directory).
    //
    // See Moler, C. & Van Loan, C. (2003). Nineteen dubious ways to
    // compute the exponential of a matrix, twenty-five years later.
    // SIAM Review, 45, 3-000.

    using namespace Scilib;
    assert(a.extent(0) == a.extent(1));

    int e = static_cast<int>(std::log2(matrix_norm(a, 'I')));
    int s = std::max(0, e + 1);

    Matrix<double, Layout> A = scaled(1.0 / std::pow(2.0, s), a);
    Matrix<double, Layout> X = A;

    double c = 0.5;

    auto E = identity<Matrix<double, Layout>>(a.extent(0)) + c * A;
    auto D = identity<Matrix<double, Layout>>(a.extent(0)) - c * A;

    const int q = 6;
    int p = 1;

    for (int k = 2; k <= q; ++k) {
        c *= (q - k + 1) / static_cast<double>(k * (2 * q - k + 1));
        X = A * X;
        auto cX = c * X;
        E += cX;
        if (p) {
            D += cX;
        }
        else {
            D -= cX;
        }
        p = !p;
    }
    E = inv(D.view()) * E;

    for (int k = 1; k <= s; ++k) {
        E = E * E;
    }
    return E;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_EXPM_H
