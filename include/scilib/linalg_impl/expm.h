// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EXPM_H
#define SCILIB_LINALG_EXPM_H

#include "auxiliary.h"
#include "matrix_norm.h"
#include <experimental/linalg>
#include <gsl/gsl>
#include <type_traits>


namespace Sci {
namespace Linalg {

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_same_v<std::remove_cv_t<T>, double>&& std::is_integral_v<IndexType>)
auto expm(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> a)
{
    // Algorithm: Matlab expm1 (demo directory).
    //
    // See Moler, C. & Van Loan, C. (2003). Nineteen dubious ways to
    // compute the exponential of a matrix, twenty-five years later.
    // SIAM Review, 45, 3-000.

    using namespace Sci;
    using namespace Sci::Linalg;

    using value_type = std::remove_cv_t<T>;

    Expects(a.extent(0) == a.extent(1));

    int e = gsl::narrow_cast<int>(std::log2(matrix_norm(a, 'I')));
    int s = std::max(0, e + 1);

    Matrix<value_type, Layout> A = std::experimental::linalg::scaled(1.0 / std::pow(2.0, s), a);
    Matrix<value_type, Layout> X = A;

    value_type c = 0.5;

    auto E = identity<Matrix<value_type, Layout>>(a.extent(0)) + c * A;
    auto D = identity<Matrix<value_type, Layout>>(a.extent(0)) - c * A;

    const int q = 6;
    int p = 1;

    for (int k = 2; k <= q; ++k) {
        c *= (q - k + 1) / gsl::narrow_cast<value_type>(k * (2 * q - k + 1));
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
    E = inv(D) * E;

    for (int k = 1; k <= s; ++k) {
        E = E * E;
    }
    return E;
}

template <class Layout>
inline Sci::Matrix<double, Layout> expm(const Sci::Matrix<double, Layout>& a)
{
    return expm(a.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_EXPM_H
