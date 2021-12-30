// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
#define SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H

#include <experimental/mdspan>
#include <cmath>
#include <type_traits>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x>
inline auto
abs_sum(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x)
{
    using size_type = stdex::extents<>::size_type;
    using value_type = std::remove_cv_t<T>;

    value_type result = 0;
    for (size_type i = 0; i < x.extent(0); ++i) {
        result += std::abs(x(i));
    }
    return result;
}

template <class T, class Layout>
inline T abs_sum(const Scilib::Vector<T, Layout>& x)
{
    return abs_sum(x.view());
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
