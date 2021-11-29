// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <armadillo>
#include <chrono>
#include <iostream>

using Timer = std::chrono::duration<double, std::milli>;

void print(int n, const Timer& t_arma, const Timer& t_eigs)
{
    std::cout << "Eigenvalues for symmetric matrix:\n"
              << "---------------------------------\n"
              << "size =      " << n << " x " << n << '\n'
              << "eigs/arma = " << t_eigs.count() / t_arma.count() << "\n\n";
}

void benchmark(int n)
{
    using namespace Scilib;
    using namespace Scilib::Linalg;

    arma::mat a1 = arma::randu<arma::mat>(n, n);
    arma::mat a2 = a1.t() * a1;
    arma::mat eigvec(n, n);
    arma::vec eigval(n);
    auto t1 = std::chrono::high_resolution_clock::now();
    arma::eig_sym(eigval, eigvec, a2);
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_arma = t2 - t1;

    Matrix<double> b1 = randu(n, n);
    Matrix<double> b1_t = b1;
    Matrix<double> b2(n, n);
    matrix_product(transposed(b1_t.view()), b1.view(), b2.view());
    Vector<double> wr(n);
    t1 = std::chrono::high_resolution_clock::now();
    eigs(b2.view(), wr.view());
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigs = t2 - t1;

    print(n, t_arma, t_eigs);
}

int main()
{
    int n = 10;
    benchmark(n);

    n = 100;
    benchmark(n);

    n = 500;
    benchmark(n);
}
