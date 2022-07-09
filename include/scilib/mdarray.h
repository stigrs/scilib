// Copyright (c) 2022 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#include "mdarray_impl/support.h"
#include <array>
#include <cstddef>
#include <experimental/mdspan>
#include <gsl/gsl>
#include <utility>
#include <valarray>
#include <vector>

#ifdef USE_MKL_ALLOCATOR
#include <scilib/mdarray_impl/mkl_allocator.h>
#define MDARRAY_ALLOCATOR(X) Sci::MKL_allocator<X>
#else
#include <memory>
#define MDARRAY_ALLOCATOR(X) std::allocator<X>
#endif

namespace stdex = std::experimental;

namespace Sci {

#ifndef SCILIB_INDEX_TYPE
#define SCILIB_INDEX_TYPE gsl::index
#endif
using index = SCILIB_INDEX_TYPE;

using layout_left = stdex::layout_left;
using layout_right = stdex::layout_right;
using layout_stride = stdex::layout_stride;

template <class ElementType,
          class Extents,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
    requires __Detail::Extents_has_rank<Extents>
class MDArray;

//--------------------------------------------------------------------------------------------------
// Stack-allocated MDArrays:

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::array<ElementType, 3>>
using Vector3 = MDArray<ElementType, stdex::extents<index, 3>, LayoutPolicy, Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::array<ElementType, 4>>
using Vector4 = MDArray<ElementType, stdex::extents<index, 4>, LayoutPolicy, Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::array<ElementType, 9>>
using Matrix33 = MDArray<ElementType, stdex::extents<index, 3, 3>, LayoutPolicy, Container>;

//--------------------------------------------------------------------------------------------------
// Heap-allocated MDArrays:

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Vector =
    MDArray<ElementType, stdex::extents<index, stdex::dynamic_extent>, LayoutPolicy, Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Matrix = MDArray<ElementType,
                       stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent>,
                       LayoutPolicy,
                       Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Array3D = MDArray<
    ElementType,
    stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
    LayoutPolicy,
    Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Array4D = MDArray<ElementType,
                        stdex::extents<index,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        LayoutPolicy,
                        Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Array5D = MDArray<ElementType,
                        stdex::extents<index,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        LayoutPolicy,
                        Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Array6D = MDArray<ElementType,
                        stdex::extents<index,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        LayoutPolicy,
                        Container>;

template <class ElementType,
          class LayoutPolicy = stdex::layout_right,
          class Container = std::vector<ElementType>>
using Array7D = MDArray<ElementType,
                        stdex::extents<index,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        LayoutPolicy,
                        Container>;

} // namespace Sci

// clang-format off
#include "mdarray_impl/mdspan_iterator.h"
#include "mdarray_impl/copy.h"
#include "mdarray_impl/copy_n.h"
#include "mdarray_impl/sort.h"
#include "mdarray_impl/swap_elements.h"
#include "mdarray_impl/slice.h"
#include "mdarray_impl/support.h"
#include "mdarray_impl/mdarray_bits.h"
#include "mdarray_impl/operations.h"
// clang-format on

#endif // SCILIB_MDARRAY_H
