// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <cmath>

Scilib::Matrix<double> Scilib::Linalg::expm(Scilib::Matrix_view<double> a)
{
    // Algorithm: Matlab expm1 (demo directory).
    //
    // See Moler, C. & Van Loan, C. (2003). Nineteen dubious ways to
    // compute the exponential of a matrix, twenty-five years later.
    // SIAM Review, 45, 3-000.

    assert(a.extent(0) == a.extent(1));

    int e = static_cast<int>(std::log2(Scilib::Linalg::matrix_norm(a, 'I')));
    int s = std::max(0, e + 1);

    Scilib::Matrix<double> A = scaled(1.0 / std::pow(2.0, s), a);
    Scilib::Matrix<double> X = A;

    double c = 0.5;

    auto E = Scilib::Linalg::identity(a.extent(0)) + c * A;
    auto D = Scilib::Linalg::identity(a.extent(0)) - c * A;

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
    E = Scilib::Linalg::inv(D.view()) * E;

    for (int k = 1; k <= s; ++k) {
        E = E * E;
    }
    return E;
}
