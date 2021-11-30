// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#include <experimental/mdspan>
#include <scilib/mdarray_impl/support.h>

#ifdef _MSC_VER
#include <concepts>
#define STD_CONVERTIBLE_TO(X) std::convertible_to<X>
#else
#define STD_CONVERTIBLE_TO(X) Scilib::__Detail::convertible_to<X>
#endif

namespace stdex = std::experimental;

namespace Scilib {

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

// clang-format off
template <class E>
concept Extents_has_rank = 
    requires (E /* exts */) { { E::rank() } -> STD_CONVERTIBLE_TO(std::size_t);
};
// clang-format on

template <class T, Extents_has_rank Extents>
/* requires Extents_has_rank<Extents> */
class MDArray;

template <class T>
using Vector = MDArray<T, stdex::extents<stdex::dynamic_extent>>;

template <class T>
using Matrix =
    MDArray<T, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>;

template <class T>
using Array3D = MDArray<T,
                        stdex::extents<stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>>;

template <class T>
using Array4D = MDArray<T,
                        stdex::extents<stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>>;

} // namespace Scilib

#include <scilib/traits.h>
#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/slice.h>
#include <scilib/mdarray_impl/operations.h>

namespace Sci = Scilib;

#endif // SCILIB_MDARRAY_H
