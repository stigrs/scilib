// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
#define SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T, class Layout, class Allocator>
inline T vector_abs_sum(const Sci::Vector<T, Layout, Allocator>& x)
{
    return std::experimental::linalg::vector_abs_sum(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_VECTOR_ABS_SUM_H
