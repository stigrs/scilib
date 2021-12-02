// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_IMPL_TRAPZ_H
#define SCILIB_INTEGRATE_IMPL_TRAPZ_H

#include <scilib/mdarray.h>

namespace Scilib {
namespace Integrate {

// Integrate function values over a non-uniform grid using the
// trapezoidal rule.
double trapz(double xlo, double xup, Scilib::Vector_view<double> y);

} // namespace Integrate
} // namespace Scilib

#endif // SCILIB_INTEGRATE_IMPL_TRAPZ_H
