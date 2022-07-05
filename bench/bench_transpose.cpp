// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5054)
#endif

#include <Eigen/Dense>
#include <chrono>
#include <experimental/linalg>
#include <iostream>
#include <scilib/mdarray.h>

using Timer = std::chrono::duration<double, std::milli>;

void print(int n, int m, const Timer& t_eigen, const Timer& t_sci)
{
    std::cout << "Matrix transpose:\n"
              << "-----------------\n"
              << "size =     " << n << " x " << m << '\n'
              << "scilib/eigen = " << t_sci.count() / t_eigen.count() << "\n\n";
}

void benchmark(int n, int m)
{
    Eigen::MatrixXd m1(n, m);
    m1.fill(1.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    m1.transpose();
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigen = t2 - t1;

    Sci::Matrix<double> m2(n, m);
    m2 = 1.0;
    t1 = std::chrono::high_resolution_clock::now();
    auto mt = std::experimental::linalg::transposed(m2.view());
    t2 = std::chrono::high_resolution_clock::now();
    (void)mt;
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
