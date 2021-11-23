// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/mdarray_impl/matrix.h>
#include <scilib/traits.h>
#include <cassert>
#include <complex>

namespace Scilib {
namespace Linalg {

inline double norm2(const Vector_view<double>& x)
{
    const Index n = narrow_cast<Index>(x.size());
    const Index incx = narrow_cast<Index>(x.stride(0));

    return cblas_dnrm2(n, x.data(), incx);
}

} // namespace Linalg
} // namespace Scilib
