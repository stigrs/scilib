// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/blas1_scale.h>
#include <cassert>

namespace Scilib {
namespace Linalg {

template <typename T>
inline Vector_view<T> scaled(const T& scalar, Vector_view<T> v)
{
    scale(scalar, v);
    return v;
}

template <typename T>
inline Matrix_view<T> scaled(const T& scalar, Matrix_view<T> m)
{
    scale(scalar, m);
    return m;
}

} // namespace Linalg
} // namespace Scilib