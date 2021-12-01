// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_TRANSPOSED_H
#define SCILIB_LINALG_TRANSPOSED_H

#include <scilib/mdarray.h>
#include <array>

namespace Scilib {
namespace Linalg {

template <class T>
inline Submatrix_view<T> transposed(Matrix_view<T> m)
{
    return Submatrix_view<T>{
        m.data(),
        {stdex::dextents<2>{m.extent(1), m.extent(0)},
         std::array<std::size_t, 2>{m.stride(1), m.stride(0)}}};
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_TRANSPOSED_H
