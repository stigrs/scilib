// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#include <experimental/mdspan>

namespace stdex = std::experimental;

namespace Scilib {

template <class T, std::size_t Rank, class Extents>
class MDArray;

template <class T>
using Vector = MDArray<T, 1, stdex::extents<stdex::dynamic_extent>>;

template <class T>
using Matrix =
    MDArray<T, 2, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>;

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

#include <scilib/traits.h>
#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/slice.h>
#include <scilib/mdarray_impl/operations.h>

namespace Sci = Scilib;

#endif // SCILIB_MDARRAY_H
