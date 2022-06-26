// Copyright (c) 2022 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_H
#define SCILIB_MDARRAY_H

#include <experimental/mdspan>
#include <scilib/mdarray_impl/support.h>
#include <cstddef>
#include <vector>
#include <utility>

#if _MSC_VER >= 1927
#include <concepts>
#define STD_CONVERTIBLE_TO(X) std::convertible_to<X>
#else
#define STD_CONVERTIBLE_TO(X) Sci::__Detail::convertible_to<X>
#endif

#ifdef USE_MKL_ALLOCATOR
#include <scilib/mdarray_impl/mkl_allocator.h>
#define MDARRAY_ALLOCATOR(X) Sci::MKL_allocator<X>
#else
#include <memory>
#define MDARRAY_ALLOCATOR(X) std::allocator<X>
#endif

namespace stdex = std::experimental;

// Signed array index.
using Index = std::ptrdiff_t;

#ifndef USE_SIGNED_EXTENTS_SIZE_TYPE
    using extents_size_type = std::size_t;
#else
    using extents_size_type = Index;
#endif


namespace Sci {

using layout_left = stdex::layout_left;
using layout_right = stdex::layout_right;
using layout_stride = stdex::layout_stride;

template <class T, class SizeType = extents_size_type, class Layout = stdex::layout_right>
using Vector_view =
    stdex::mdspan<T, stdex::extents<SizeType, stdex::dynamic_extent>, Layout>;

template <class T>
using Subvector_view = stdex::
    mdspan<T, stdex::extents<Size_type, stdex::dynamic_extent>, stdex::layout_stride>;

template <class T, class Layout = stdex::layout_right>
using Matrix_view =
    stdex::mdspan<T,
                  stdex::extents<Size_type, stdex::dynamic_extent, stdex::dynamic_extent>,
                  Layout>;

template <class T>
using Submatrix_view =
    stdex::mdspan<T,
                  stdex::extents<Size_type, stdex::dynamic_extent, stdex::dynamic_extent>,
                  stdex::layout_stride>;

// clang-format off
template <class E>
concept Extents_has_rank = 
    requires (E /* exts */) { { E::rank() } -> STD_CONVERTIBLE_TO(std::size_t);
};

template <class M>
concept MDArray_type = 
    requires (M /* m */) { { M::rank() } -> STD_CONVERTIBLE_TO(std::size_t);
};

template <class T, 
          class Extents, 
          class Layout = stdex::layout_right, 
          class Allocator = MDARRAY_ALLOCATOR(T)> 
    requires Extents_has_rank<Extents> 
class MDArray;
// clang-format on

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Vector =
    MDArray<T, stdex::extents<Size_type, 
                              stdex::dynamic_extent>, Layout, Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Matrix =
    MDArray<T,
            stdex::extents<Size_type, stdex::dynamic_extent, stdex::dynamic_extent>,
            Layout,
            Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Array3D = MDArray<T,
                        stdex::extents<Size_type, 
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        Layout,
                        Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Array4D = MDArray<T,
                        stdex::extents<Size_type, 
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        Layout,
                        Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Array5D = MDArray<T,
                        stdex::extents<Size_type, 
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        Layout,
                        Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Array6D = MDArray<T,
                        stdex::extents<Size_type, 
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        Layout,
                        Allocator>;

template <class T,
          class Layout = stdex::layout_right,
          class Allocator = MDARRAY_ALLOCATOR(T)>
using Array7D = MDArray<T,
                        stdex::extents<Size_type, 
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent,
                                       stdex::dynamic_extent>,
                        Layout,
                        Allocator>;

} // namespace Sci

#include <scilib/mdarray_impl/mdspan_iterator.h>
#include <scilib/mdarray_impl/copy.h>
#include <scilib/mdarray_impl/copy_n.h>
#include <scilib/mdarray_impl/sort.h>
#include <scilib/mdarray_impl/swap_elements.h>
#include <scilib/mdarray_impl/slice.h>
#include <scilib/mdarray_impl/support.h>
#include <scilib/mdarray_impl/mdarray_bits.h>
#include <scilib/mdarray_impl/operations.h>

#endif // SCILIB_MDARRAY_H
