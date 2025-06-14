// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_SOLVE_IVP_H
#define SCILIB_INTEGRATE_SOLVE_IVP_H

#include "../mdarray.h"
#include <cassert>
#include <cmath>
#include <exception>
#include <limits>

namespace Sci {
namespace Integrate {
namespace __Detail {

// Compute error norm.
template <class IndexType, std::size_t ext, class Layout, class Container>
double
error_norm(const Sci::MDArray<double, Mdspan::extents<IndexType, ext>, Layout, Container>& y,
           const Sci::MDArray<double, Mdspan::extents<IndexType, ext>, Layout, Container>& ynew,
           const Sci::MDArray<double, Mdspan::extents<IndexType, ext>, Layout, Container>& err_vec,
           double atol,
           double rtol)
{
    using index_type = IndexType;

    assert(y.size() == ynew.size());
    assert(y.size() == err_vec.size());

    double max_norm = std::abs(err_vec[0]) / (atol + std::max(std::abs(y[0]), std::abs(ynew[0])) * rtol);
    for (index_type i = 1; i < err_vec.extent(0); ++i) {
        double tol = atol + std::max(std::abs(y[i]), std::abs(ynew[i])) * rtol;
        double val = std::abs(err_vec[i]) / tol;
        if (val > max_norm) {
            max_norm = val;
        }
    }
    return max_norm;
}

// Embedded Dormand-Prince method of order 4(5).
template <class F, class IndexType, std::size_t ext, class Layout, class Container>
void dormand_prince(F f,
                    double& x,
                    double xf,
                    Sci::MDArray<double, Mdspan::extents<IndexType, ext>, Layout, Container>& y,
                    double atol,
                    double rtol)
{
    // Butcher tableau for Dormand-Prince method:
    // Source: https://en.wikipedia.org/wiki/Dormand-Prince_method

    constexpr double c1 = 0.0;
    constexpr double c2 = 1.0 / 5.0;
    constexpr double c3 = 3.0 / 10.0;
    constexpr double c4 = 4.0 / 5.0;
    constexpr double c5 = 8.0 / 9.0;
    constexpr double c6 = 1.0;
    constexpr double c7 = 1.0;

    constexpr double a21 = 1.0 / 5.0;
    constexpr double a31 = 3.0 / 40.0;
    constexpr double a32 = 9.0 / 40.0;
    constexpr double a41 = 44.0 / 45.0;
    constexpr double a42 = -56.0 / 15.0;
    constexpr double a43 = 32.0 / 9.0;
    constexpr double a51 = 19372.0 / 6561.0;
    constexpr double a52 = -25360.0 / 2187.0;
    constexpr double a53 = 64448.0 / 6561.0;
    constexpr double a54 = -212.0 / 729.0;
    constexpr double a61 = 9017.0 / 3168.0;
    constexpr double a62 = -355.0 / 33.0;
    constexpr double a63 = 46732.0 / 5247.0;
    constexpr double a64 = 49.0 / 176.0;
    constexpr double a65 = -5103.0 / 18656.0;
    constexpr double a71 = 35.0 / 384.0;
    constexpr double a72 = 0.0;
    constexpr double a73 = 500.0 / 1113.0;
    constexpr double a74 = 125.0 / 192.0;
    constexpr double a75 = -2187.0 / 6784.0;
    constexpr double a76 = 11.0 / 84.0;

    constexpr double b1 = 35.0 / 384.0;
    constexpr double b2 = 0.0;
    constexpr double b3 = 500.0 / 1113.0;
    constexpr double b4 = 125.0 / 192.0;
    constexpr double b5 = -2187.0 / 6784.0;
    constexpr double b6 = 11.0 / 84.0;
    constexpr double b7 = 0.0;

    constexpr double bs1 = 5179.0 / 57600.0;
    constexpr double bs2 = 0.0;
    constexpr double bs3 = 7571.0 / 16695.0;
    constexpr double bs4 = 393.0 / 640.0;
    constexpr double bs5 = -92097.0 / 339200.0;
    constexpr double bs6 = 187.0 / 2100.0;
    constexpr double bs7 = 1.0 / 40.0;

    constexpr double e1 = b1 - bs1;
    constexpr double e2 = b2 - bs2;
    constexpr double e3 = b3 - bs3;
    constexpr double e4 = b4 - bs4;
    constexpr double e5 = b5 - bs5;
    constexpr double e6 = b6 - bs6;
    constexpr double e7 = b7 - bs7;

    constexpr double safety = 0.9;
    constexpr double max_scale = 2.0;
    constexpr double min_scale = 0.3;

    const double hmax = safety * (xf - x);
    const double hmin = 1.0e-12 * hmax;
    double h = hmax;

    constexpr int max_iter = 100;
    int iter = 0;

    // Algorithm: Runge-Kutta-Fehlberg method from Wikipedia
    while (x < xf) {
        if (h < hmin) {
            h = hmin;
        }
        if (h > hmax) {
            h = hmax;
        }
        if (x + h > xf) {
            h = xf - x;
        }
        // clang-format off
        auto k1 = f(x + c1 * h, y);
        auto k2 = f(x + c2 * h, y + h * (a21 * k1));
        auto k3 = f(x + c3 * h, y + h * (a31 * k1 + a32 * k2));
        auto k4 = f(x + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
        auto k5 = f(x + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
        auto k6 = f(x + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));
        auto k7 = f(x + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6));
        auto err_vec = h * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);
        auto ynew = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7);
        // clang-format on
        double error_norm = Sci::Integrate::__Detail::error_norm(y, ynew, err_vec, atol, rtol);

        if (error_norm > 1.0) { // reject the step
            double scale = safety * std::pow(1.0 / error_norm, 0.2);
            h *= std::min(std::max(scale, min_scale), max_scale);
            ++iter;
        }
        else { // accept the step
            x += h;
            y = ynew;
            if (error_norm <= std::numeric_limits<double>::epsilon()) {
                h *= max_scale; // error too small; increase step size
            }
        }
        if (iter > max_iter) {
            std::runtime_error("dormand_prince failed to converge");
        }
    }
}

} // namespace __Detail

// Solve an inital value problem for a system of ODEs.
//
// dy/dx = f(x, y)
// y(x0) = y0
//
template <class F, class IndexType, std::size_t ext, class Layout, class Container>
inline void solve_ivp(F f,
                      double& x,
                      double xf,
                      Sci::MDArray<double, Mdspan::extents<IndexType, ext>, Layout, Container>& y,
                      double atol = 1.0e-7,
                      double rtol = 1.0e-7)
{
    __Detail::dormand_prince(f, x, xf, y, atol, rtol);
}

} // namespace Integrate
} // namespace Sci

#endif // SCILIB_INTEGRATE_SOLVE_IVP_H
