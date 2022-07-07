// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifdef USE_MKL
#include <mkl.h>

#include <iostream>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

void benchmark(BLAS_INT m, BLAS_INT n)
{
    double time_start = dsecnd();

    const BLAS_INT k = n;
    const BLAS_INT lda = k;
    const BLAS_INT ldb = n;
    const BLAS_INT ldc = n;

    Sci::Matrix<double> A(m, k);
    Sci::Matrix<double> B(k, n);
    Sci::Matrix<double> C(m, n);

    A = 1.0;
    B = 1.0;

    int loop_count = 100;

    time_start = dsecnd();
    for (int it = 0; it < loop_count; ++it) {
        Sci::Linalg::matrix_product(A, B, C);
    }
    double time_end = dsecnd();
    double time_avg = (time_end - time_start) / loop_count;
    double gflop = (2.0 * m * n * k) * 1.0e-9;

    std::cout << "M x N = " << m << " x " << n << '\n'
              << "Gflops/sec (scilib): " << gflop / time_avg << '\n';

    time_start = dsecnd();

    double* MKL_A = (double*) mkl_malloc(sizeof(double) * m * k, 64);
    double* MKL_B = (double*) mkl_malloc(sizeof(double) * k * n, 64);
    double* MKL_C = (double*) mkl_malloc(sizeof(double) * m * n, 64);

    for (BLAS_INT i = 0; i < m * k; ++i) {
        MKL_A[i] = 1.0;
    }
    for (BLAS_INT i = 0; i < k * n; ++i) {
        MKL_B[i] = 1.0;
    }
    const double alpha = 1.0;
    const double beta = 0.0;

    time_start = dsecnd();
    for (int it = 0; it < loop_count; ++it) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, MKL_A, lda, MKL_B,
                    ldb, beta, MKL_C, ldc);
    }
    time_end = dsecnd();
    time_avg = (time_end - time_start) / loop_count;

    std::cout << "Gflops/sec (MKL):    " << gflop / time_avg << "\n\n";

    mkl_free(MKL_A);
    mkl_free(MKL_B);
    mkl_free(MKL_C);
}
#endif

int main()
{
#ifdef USE_MKL
    BLAS_INT m = 10;
    BLAS_INT n = 5;
    benchmark(m, n);

    m = 100;
    n = 50;
    benchmark(m, n);

    m = 1000;
    n = 500;
    benchmark(m, n);

    m = 2000;
    n = 2048;
    benchmark(m, n);
#endif
}
