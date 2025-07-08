// Copyright (c) 2022 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#ifdef MDSPAN_USE_PAREN_OPERATOR
#undef MDSPAN_USE_PAREN_OPERATOR
#endif
// Needed for stdBLAS:
#define MDSPAN_USE_PAREN_OPERATOR 1

#include "mdarray_impl/support.h"
#include <array>
#include <cstddef>
#include <mdspan/mdspan.hpp>
#include <experimental/__p2630_bits/submdspan.hpp>
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

namespace Sci { 
#ifndef SCILIB_INDEX_TYPE
#define SCILIB_INDEX_TYPE gsl::index
#endif
using index = SCILIB_INDEX_TYPE;

using layout_left = Kokkos::layout_left;
using layout_right = Kokkos::layout_right;
using layout_stride = Kokkos::layout_stride;

template <class ElementType,
          class Extents,
          class LayoutPolicy = Kokkos::layout_right,
          class Container = std::vector<ElementType>>
    requires __Detail::Is_extents_v<Extents>
class MDArray;

//--------------------------------------------------------------------------------------------------
// Stack-allocated MDArrays:

template <class ElementType, std::size_t ext, class LayoutPolicy = Kokkos::layout_right>
using StaticVector =
    MDArray<ElementType, Kokkos::extents<index, ext>, LayoutPolicy, std::array<ElementType, ext>>;

template <class ElementType,
          std::size_t nrows,
          std::size_t ncols,
          class LayoutPolicy = Kokkos::layout_right>
using StaticMatrix = MDArray<ElementType,
                             Kokkos::extents<index, nrows, ncols>,
                             LayoutPolicy,
                             std::array<ElementType, nrows * ncols>>;

//--------------------------------------------------------------------------------------------------
// Heap-allocated MDArrays:

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Vector = MDArray<ElementType,
                       Kokkos::extents<index, Kokkos::dynamic_extent>,
                       LayoutPolicy,
                       std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Matrix = MDArray<ElementType,
                       Kokkos::extents<index, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
                       LayoutPolicy,
                       std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Array3D = MDArray<
    ElementType,
    Kokkos::extents<index, Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
    LayoutPolicy,
    std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Array4D = MDArray<ElementType,
                        Kokkos::extents<index,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent>,
                        LayoutPolicy,
                        std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Array5D = MDArray<ElementType,
                        Kokkos::extents<index,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent>,
                        LayoutPolicy,
                        std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Array6D = MDArray<ElementType,
                        Kokkos::extents<index,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent>,
                        LayoutPolicy,
                        std::vector<ElementType>>;

template <class ElementType, class LayoutPolicy = Kokkos::layout_right>
using Array7D = MDArray<ElementType,
                        Kokkos::extents<index,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent,
                                       Kokkos::dynamic_extent>,
                        LayoutPolicy,
                        std::vector<ElementType>>;

} // namespace Sci

// clang-format off
#include "mdarray_impl/mdspan_iterator.h"
#include "mdarray_impl/for_each_in_extents.h"
#include "mdarray_impl/copy.h"
#include "mdarray_impl/copy_n.h"
#include "mdarray_impl/sort.h"
#include "mdarray_impl/swap_elements.h"
#include "mdarray_impl/slice.h"
#include "mdarray_impl/mdarray_bits.h"
#include "mdarray_impl/operations.h"
// clang-format on

#endif // SCILIB_MDARRAY_H
