// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_H
#define SCILIB_LINALG_H

// Do not change this ordering.

#include "mdarray.h"

#include "linalg_impl/lapack_types.h"
#include "linalg_impl/auxiliary.h"
#include "linalg_impl/element_wise_math.h"

#include "linalg_impl/blas1_add.h"
#include "linalg_impl/blas1_axpy.h"
#include "linalg_impl/blas1_dot.h"
#include "linalg_impl/blas1_idx_abs_max.h"
#include "linalg_impl/blas1_idx_abs_min.h"
#include "linalg_impl/blas1_scale.h"
#include "linalg_impl/blas1_vector_abs_sum.h"
#include "linalg_impl/blas1_vector_norm2.h"
#include "linalg_impl/blas2_matrix_vector_product.h"
#include "linalg_impl/blas3_matrix_product.h"

#include "linalg_impl/scaled.h"
#include "linalg_impl/trace.h"
#include "linalg_impl/transposed.h"
#include "linalg_impl/matrix_decomposition.h"
#include "linalg_impl/matrix_norm.h"
#include "linalg_impl/det.h"
#include "linalg_impl/eigenvalue.h"
#include "linalg_impl/inv.h"
#include "linalg_impl/expm.h"
#include "linalg_impl/matrix_power.h"
#include "linalg_impl/linsolve.h"
#include "linalg_impl/lstsq.h"

#endif // SCILIB_LINALG_H
