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
#include <iostream>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <valarray>

typedef std::chrono::duration<double, std::milli> Timer;

void print(int n,
           const Timer& t_eigen,
           const Timer& t_sci,
           const Timer& t_val,
           const Timer& t_loop,
           const Timer& t_axpy)
{
    std::cout << "Vector addition:\n"
              << "----------------\n"
              << "size =         " << n << '\n'
              << "scilib/eigen = " << t_sci.count() / t_eigen.count() << "\n"
              << "scilib/val =   " << t_sci.count() / t_val.count() << "\n"
              << "scilib/loop =  " << t_sci.count() / t_loop.count() << "\n"
              << "axpy/eigen =   " << t_axpy.count() / t_eigen.count() << "\n\n";
}

void benchmark(int n)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    Eigen::VectorXd aa(n);
    Eigen::VectorXd ab(n);

    aa.fill(1.0);
    ab.fill(1.0);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        ab = 2.0 * aa + ab;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigen = t2 - t1;

    Vector<double> va(n);
    Vector<double> vb(n);

    va = 1.0;
    vb = 1.0;

    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        vb = 2.0 * va + vb;
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_sci = t2 - t1;

    va = 1.0;
    vb = 1.0;

    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        for (std::size_t i = 0; i < vb.size(); ++i) {
            vb(i) = 2.0 * va(i) + vb(i);
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_loop = t2 - t1;

    va = 1.0;
    vb = 1.0;

    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        Kokkos::Experimental::linalg::add(Kokkos::Experimental::linalg::scaled(2.0, va.to_mdspan()), vb.to_mdspan(),
                                       vb.to_mdspan());
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_axpy = t2 - t1;

    std::valarray<double> wa(1.0, n);
    std::valarray<double> wb(1.0, n);
    t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        wb = 2.0 * wa + wb;
    }
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_val = t2 - t1;

    print(n, t_eigen, t_sci, t_val, t_loop, t_axpy);
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

#ifdef _MSC_VER
#pragma warning(pop)
#endif
