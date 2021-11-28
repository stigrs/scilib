// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_TRAITS_H
#define SCILIB_TRAITS_H

#include <utility>

//------------------------------------------------------------------------------
// Define integer type for use with BLAS, LAPACK and Intel MKL.

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_INT MKL_INT
#else
#define BLAS_INT int
#endif

//------------------------------------------------------------------------------
// Type cast:

// A searchable way to do narrowing casts of values.
template <typename T, typename U>
constexpr T narrow_cast(U&& u)
{
    return static_cast<T>(std::forward<U>(u));
}

#endif // SCILIB_TRAITS_H
