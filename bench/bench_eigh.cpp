// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4305)
#pragma warning(disable : 5054)
#endif

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

using Timer = std::chrono::duration<double, std::milli>;

void print(int n, const Timer& t_eigen, const Timer& t_sci)
{
    std::cout << "Eigenvalues for symmetric matrix:\n"
              << "---------------------------------\n"
              << "size =      " << n << " x " << n << '\n'
              << "sci/eigen = " << t_sci.count() / t_eigen.count() << "\n\n";
}

void benchmark(int n)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    Eigen::MatrixXd a1 = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd a2 = a1 + a1.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    auto t1 = std::chrono::high_resolution_clock::now();
    es.compute(a2);
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigen = t2 - t1;

    Matrix<double> b1 = randu<Matrix<double>>(n, n);
    Matrix<double> b1_t = b1;
    Matrix<double> b2(n, n);
    matrix_product(transposed(b1_t), b1, b2);
    Vector<double> wr(n);
    t1 = std::chrono::high_resolution_clock::now();
    eigh(b2.view(), wr.view());
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_sci = t2 - t1;

    print(n, t_eigen, t_sci);
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

#ifdef _MSC_VER
#pragma warning(pop)
#endif
