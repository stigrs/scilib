// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/linalg_impl/blas1_scale.h>
#include <experimental/mdspan>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor>
scaled(const T& scalar,
       stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v)
{
    scale(scalar, v);
    return v;
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor>
scaled(const T& scalar,
       stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m)
{
    scale(scalar, m);
    return m;
}

} // namespace Linalg
} // namespace Scilib
