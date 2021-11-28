// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_TYPE_ALIASES_H
#define SCILIB_MDARRAY_TYPE_ALIASES_H

#include <experimental/mdspan>

namespace Scilib {
namespace stdex = std::experimental;

template <typename T>
using Vector_view = stdex::mdspan<T, stdex::extents<stdex::dynamic_extent>>;

template <typename T>
using Subvector_view = stdex::
    mdspan<T, stdex::extents<stdex::dynamic_extent>, stdex::layout_stride>;

template <typename T>
using Matrix_view =
    stdex::mdspan<T,
                  stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>,
                  stdex::layout_right>;

template <typename T>
using Submatrix_view =
    stdex::mdspan<T,
                  stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>,
                  stdex::layout_stride>;

} // namespace Scilib

#endif // SCILIB_MDARRAY_TYPE_ALIASES_H
