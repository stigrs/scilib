// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_TRANSPOSED_H
#define SCILIB_LINALG_TRANSPOSED_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T, class Layout, class Container>
inline Sci::Matrix<T, Layout, Container> transposed(const Sci::Matrix<T, Layout, Container>& a)
{
    return Sci::Matrix<T, Layout, Container>(std::experimental::linalg::transposed(a.view()));
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_TRANSPOSED_H
