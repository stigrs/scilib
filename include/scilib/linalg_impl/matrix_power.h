// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_POWER_H
#define SCILIB_LINALG_MATRIX_POWER_H

namespace Sci {
namespace Linalg {

#include <cmath>
#include <type_traits>

// Raise a square matrix to the (integer) power n.
template <class T, class Layout>
inline auto matrix_power(Sci::Matrix_view<T, Layout> m, int n)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    using value_type = std::remove_cv_t<T>;

    assert(m.extent(0) == m.extent(1));

    Matrix<value_type, Layout> tmp(m);

    if (n < 0) {
        inv(tmp.view(), tmp.view());
    }
    int nn = std::abs(n);

    Matrix<value_type, Layout> res(m.extent(0), m.extent(1));

    if (nn == 0) {
        res = identity<Matrix<value_type, Layout>>(m.extent(0));
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

template <class T, class Layout, class Allocator>
inline Sci::Matrix<T, Layout, Allocator> matrix_power(const Sci::Matrix<T, Layout, Allocator>& m,
                                                      int n)
{
    return matrix_power(m.view(), n);
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_POWER_H
