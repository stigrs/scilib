// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_H
#define SCILIB_LINALG_H

// Do not change this ordering.

#include <scilib/mdarray.h>

#include <scilib/linalg_impl/lapack_types.h>
#include <scilib/linalg_impl/auxiliary.h>
#include <scilib/linalg_impl/element_wise_math.h>
#include <scilib/linalg_impl/sort.h>

#include <scilib/linalg_impl/blas1_add.h>
#include <scilib/linalg_impl/blas1_axpy.h>
#include <scilib/linalg_impl/blas1_dot.h>
#include <scilib/linalg_impl/blas1_idx_abs_max.h>
#include <scilib/linalg_impl/blas1_idx_abs_min.h>
#include <scilib/linalg_impl/blas1_scale.h>
#include <scilib/linalg_impl/blas1_vector_abs_sum.h>
#include <scilib/linalg_impl/blas1_vector_norm2.h>
#include <scilib/linalg_impl/blas2_matrix_vector_product.h>
#include <scilib/linalg_impl/blas3_matrix_product.h>

#include <scilib/linalg_impl/scaled.h>
#include <scilib/linalg_impl/transposed.h>
#include <scilib/linalg_impl/matrix_decomposition.h>
#include <scilib/linalg_impl/matrix_norm.h>
#include <scilib/linalg_impl/det.h>
#include <scilib/linalg_impl/eigenvalue.h>
#include <scilib/linalg_impl/inv.h>
#include <scilib/linalg_impl/expm.h>
#include <scilib/linalg_impl/matrix_power.h>
#include <scilib/linalg_impl/linsolve.h>
#include <scilib/linalg_impl/lstsq.h>

#endif // SCILIB_LINALG_H
