// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_VECTOR_NORM2_H
#define SCILIB_LINALG_BLAS1_VECTOR_NORM2_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T, class Layout, class Container>
inline T vector_norm2(const Sci::Vector<T, Layout, Container>& x)
{
    return std::experimental::linalg::vector_norm2(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_VECTOR_NORM2_H
