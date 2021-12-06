// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_POWER_H
#define SCILIB_LINALG_MATRIX_POWER_H

namespace Scilib {
namespace Linalg {

#include <scilib/mdarray.h>
#include <cmath>

// Raise a square matrix to the (integer) power n.
template <class T>
inline Scilib::Matrix<T> matrix_power(Scilib::Matrix_view<T> m, int n)
{
    using namespace Scilib;

    assert(m.extent(0) == m.extent(1));

    Scilib::Matrix<T> tmp(m);

    if (n < 0) {
        inv(tmp.view(), tmp.view());
    }
    int nn = std::abs(n);

    Scilib::Matrix<T> res(m.extent(0), m.extent(1));

    if (nn == 0) {
        res = identity(m.extent(0));
    }
    else if (nn == 1) {
        res = tmp;
    }
    else if (nn == 2) {
#ifdef USE_MKL
        matrix_product(tmp.view(), tmp.view(), res.view());
#else
        res = tmp * tmp;
#endif
    }
    else {
        res = tmp;
        for (int ni = 1; ni < nn; ++ni) {
#ifdef USE_MKL
            matrix_product(res.view(), tmp.view(), res.view());
#else
            res = res * tmp;
#endif
        }
    }
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_MATRIX_POWER_H
