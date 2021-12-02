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

void print(int n, int m, const Timer& t_arma, const Timer& t_sci)
{
    std::cout << "Matrix transpose:\n"
              << "-----------------\n"
              << "size =     " << n << " x " << m << '\n'
              << "scilib/arma = " << t_sci.count() / t_arma.count() << "\n\n";
}

void benchmark(int n, int m)
{
    arma::mat m1 = arma::ones<arma::mat>(n, m);
    auto t1 = std::chrono::high_resolution_clock::now();
    m1.t();
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_arma = t2 - t1;

    Scilib::Matrix<double> m2(n, m);
    m2 = 1.0;
    t1 = std::chrono::high_resolution_clock::now();
    auto mt = Scilib::Linalg::transposed(m2.view());
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_sci = t2 - t1;

    print(n, m, t_arma, t_sci);
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
