// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray_impl/type_aliases.h>
#include <array>

namespace Scilib {
namespace Linalg {

template <typename T>
Submatrix_view<T> transposed(Matrix_view<T> m)
{
    return Submatrix_view<T>{m.data(), {stdex::dextents<2>{m.extent(1), m.extent(0)}, std::array<std::size_t, 2>{m.stride(1), m.stride(0)}}};
//    return Submatrix_view<T>(
//        m.data(), {{m.extent(1), m.extent(0)}, {m.stride(1), m.stride(0)}});
}

} // namespace Linalg
} // namespace Scilib
