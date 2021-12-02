// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EXPM_H
#define SCILIB_LINALG_EXPM_H

#include <scilib/mdarray.h>

namespace Scilib {
namespace Linalg {

Scilib::Matrix<double> expm(Scilib::Matrix_view<double> a);

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_EXPM_H
