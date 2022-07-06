// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_SCALE_H
#define SCILIB_LINALG_BLAS1_SCALE_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T, class Extents, class Layout, class Allocator>
inline void scale(const T& scalar, Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    static_assert(m.rank() <= 2);
    std::experimental::linalg::scale(scalar, m.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_SCALE_H
