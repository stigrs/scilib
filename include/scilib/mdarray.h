// Copyright (c) 2022 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#include <cstddef>
#include <experimental/mdarray>
#include <experimental/mdspan>
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
#define SCILIB_INDEX_TYPE std::size_t
#endif
using index = SCILIB_INDEX_TYPE;

using layout_left = stdex::layout_left;
using layout_right = stdex::layout_right;
using layout_stride = stdex::layout_stride;

template <class T, class Layout = stdex::layout_right>
using Vector_view = stdex::mdspan<T, stdex::extents<index, stdex::dynamic_extent>, Layout>;

template <class T>
using Subvector_view =
    stdex::mdspan<T, stdex::extents<index, stdex::dynamic_extent>, stdex::layout_stride>;

template <class T, class Layout = stdex::layout_right>
using Matrix_view =
    stdex::mdspan<T, stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent>, Layout>;

template <class T>
using Submatrix_view =
    stdex::mdspan<T,
                  stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent>,
                  stdex::layout_stride>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Vector = stdex::mdarray<T, stdex::extents<index, stdex::dynamic_extent>, Layout, Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Matrix = stdex::mdarray<T,
                              stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent>,
                              Layout,
                              Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Array3D = stdex::mdarray<
    T,
    stdex::extents<index, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>,
    Layout,
    Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Array4D = stdex::mdarray<T,
                               stdex::extents<index,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent>,
                               Layout,
                               Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Array5D = stdex::mdarray<T,
                               stdex::extents<index,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent>,
                               Layout,
                               Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Array6D = stdex::mdarray<T,
                               stdex::extents<index,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent>,
                               Layout,
                               Container>;

template <class T,
          class Layout = stdex::layout_right,
          class Container = std::vector<T, MDARRAY_ALLOCATOR(T)>>
using Array7D = stdex::mdarray<T,
                               stdex::extents<index,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent,
                                              stdex::dynamic_extent>,
                               Layout,
                               Container>;

} // namespace Sci

// clang-format off
#include "mdarray_impl/mdspan_iterator.h"
#include "mdarray_impl/copy.h"
//#include "mdarray_impl/copy_n.h"
//#include "mdarray_impl/sort.h"
//#include "mdarray_impl/swap_elements.h"
//#include "mdarray_impl/slice.h"
#include "mdarray_impl/mdarray_ext.h"
//#include "mdarray_impl/operations.h"
// clang-format on

#endif // SCILIB_MDARRAY_H
