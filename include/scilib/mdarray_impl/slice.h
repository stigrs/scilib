// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SLICE_H
#define SCILIB_MDARRAY_SLICE_H

//#include <scilib/mdarray_impl/type_aliases.h>
#include <experimental/mdspan>
#include <array>
#include <utility>

namespace Scilib {
namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline auto
row(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
    stdex::extents<>::size_type i)
{
    return stdex::submdspan(m, i, stdex::full_extent);
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline auto
column(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
       stdex::extents<>::size_type j)
{
    return stdex::submdspan(m, stdex::full_extent, j);
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto diag(stdex::mdspan<T, stdex::extents<ext, ext>, Layout, Accessor> m)
{

    return Subvector_view<T>{m.data(),
                             {stdex::dextents<1>{m.extent(0)},
                              std::array<std::size_t, 1>{m.stride(0) + 1}}};
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline auto submatrix(
    stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
    const std::pair<stdex::extents<>::size_type, stdex::extents<>::size_type>&
        row_slice,
    const std::pair<stdex::extents<>::size_type, stdex::extents<>::size_type>&
        col_slice)
{
    using size_type = stdex::extents<>::size_type;
    return stdex::submdspan(m, row_slice, col_slice);
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_SLICE_H