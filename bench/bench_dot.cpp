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
#include <valarray>
#include <numeric>

using Timer = std::chrono::duration<double, std::micro>;

void print(int n, const Timer& t_arma, const Timer& t_sci, const Timer& t_val)
{
    std::cout << "Dot product:\n"
              << "------------\n"
              << "size =            " << n << '\n'
              << "scilib/arma =     " << t_sci.count() / t_arma.count() << "\n"
              << "scilib/valarray = " << t_sci.count() / t_val.count()
              << "\n\n";
}

void benchmark(int n)
{
    arma::vec aa(n);
    arma::vec ab(n);
    aa.fill(1.0);
    ab.fill(2.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dot_arma;
    for (int it = 0; it < 10; ++it) {
        dot_arma = arma::dot(aa, ab);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_arma = t2 - t1;
    (void) dot_arma; // ignore unused result

    Sci::Vector<double> na(n);
    Sci::Vector<double> nb(n);
    na = 1.0;
    nb = 2.0;
    t1 = std::chrono::high_resolution_clock::now();
    double num;
    for (int it = 0; it < 10; ++it) {
        num = Sci::dot_product(na.view(), nb.view());
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_sci = t2 - t1;
    (void) num;

    std::valarray<double> va(1.0, n);
    std::valarray<double> vb(2.0, n);
    t1 = std::chrono::high_resolution_clock::now();
    double val;
    for (int it = 0; it < 10; ++it) {
        val = std::inner_product(std::begin(va), std::end(va), std::begin(vb),
                                 0.0);
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_val = t2 - t1;
    (void) val;

    print(n, t_arma, t_sci, t_val);
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
