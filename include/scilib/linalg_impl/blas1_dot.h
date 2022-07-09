// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_DOT_H
#define SCILIB_LINALG_BLAS1_DOT_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T, class Layout_x, class Container_x, class Layout_y, class Container_y>
inline T dot(const Sci::Vector<T, Layout_x, Container_x>& x,
             const Sci::Vector<T, Layout_y, Container_y>& y)
{
    return std::experimental::linalg::dot(x.view(), y.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_DOT_H
