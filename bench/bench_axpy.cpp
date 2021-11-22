// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <armadillo>
#include <chrono>
#include <iostream>
#include <valarray>

typedef std::chrono::duration<double, std::milli> Timer;

void print(int n,
           const Timer& t_arma,
           const Timer& t_scilib,
           const Timer& t_val)
{
    std::cout << "Vector addition:\n"
              << "----------------\n"
              << "size =        " << n << '\n'
              << "scilib/arma = " << t_scilib.count() / t_arma.count() << "\n"
              << "scilib/val =  " << t_scilib.count() / t_val.count() << "\n\n";
}

void benchmark(int n)
{
    arma::vec aa(n);
    arma::vec ab(n);
    aa.fill(1.0);
    ab.fill(1.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10; ++it) {
        ab = 2.0 * aa + ab;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_arma = t2 - t1;

    Sci::Vector<double> va(n);
    Sci::Vector<double> vb(n);

    va = 1.0;
    vb = 1.0;

    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10; ++it) {
        vb = 2.0 * va + vb;
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_scilib = t2 - t1;

    va = 1.0;
    vb = 1.0;

    std::valarray<double> wa(1.0, n);
    std::valarray<double> wb(1.0, n);
    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10; ++it) {
        wb = 2.0 * wa + wb;
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_val = t2 - t1;

    print(n, t_arma, t_scilib, t_val);
}

int main()
{
    int n = 10;
    benchmark(n);

    n = 100;
    benchmark(n);

    n = 1000;
    benchmark(n);

    n = 10000;
    benchmark(n);

    n = 100000;
    benchmark(n);
}
