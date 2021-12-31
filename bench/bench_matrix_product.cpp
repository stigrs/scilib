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

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>

using Timer = std::chrono::duration<double, std::milli>;

void print(int n, int m, const Timer& t_eigen, const Timer& t_sci)
{
    std::cout << "Matrix-matrix multiplication:\n"
              << "-----------------------------\n"
              << "size =        " << n << " x " << m << '\n'
              << "scilib/eigen = " << t_sci.count() / t_eigen.count() << "\n\n";
}

void benchmark(int n, int m)
{
    Eigen::MatrixXd a1(n, m);
    Eigen::MatrixXd a2(m, n);
    a1.fill(1.0);
    a2.fill(1.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10; ++it) {
        Eigen::MatrixXd a3 = a1 * a2;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigen = t2 - t1;

    Sci::Matrix<double> b1(n, m);
    b1 = 1.0;
    Sci::Matrix<double> b2(m, n);
    b2 = 1.0;
    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10; ++it) {
        auto b3 = b1 * b2;
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_sci = t2 - t1;

    print(n, m, t_eigen, t_sci);
}

int main()
{
    int n = 10;
    int m = 5;
    benchmark(n, m);

    n = 100;
    m = 50;
    benchmark(n, m);

    n = 1000;
    m = 500;
    benchmark(n, m);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
