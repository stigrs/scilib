// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_ADD_H
#define SCILIB_LINALG_BLAS1_ADD_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T_x,
          class Layout_x,
          class Allocator_x,
          class T_y,
          class Layout_y,
          class Allocator_y,
          class T_z,
          class Layout_z,
          class Allocator_z>
    requires(!std::is_const_v<T_z>)
inline void add(const Sci::Vector<T_x, Layout_x, Allocator_x>& x,
                const Sci::Vector<T_y, Layout_y, Allocator_y>& y,
                Sci::Vector<T_z, Layout_z, Allocator_z>& z)
{
    std::experimental::linalg::add(x.view(), y.view(), z.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_ADD_H
