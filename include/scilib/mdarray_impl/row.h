// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_ROW_H
#define SCILIB_MDARRAY_ROW_H

#include <experimental/mdspan>

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

} // namespace Scilib

#endif // SCILIB_MDARRAY_ROW_H
