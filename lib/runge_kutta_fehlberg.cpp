// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/integrate.h>
#include <scilib/linalg.h>
#include <exception>

void Scilib::Integrate::__Detail::runge_kutta_fehlberg(
    std::function<Scilib::Vector<double>(double, const Scilib::Vector<double>&)>
        f,
    double& x,
    double xf,
    Scilib::Vector<double>& y,
    double tol,
    double hmin)
{
    // Butcher tableau for Fehlberg's 4(5) method:
    // Source: https://en.wikipedia.org/wiki/Runge-Kutta-Fehlberg_method

    constexpr double c1 = 0.0;
    constexpr double c2 = 1.0 / 4.0;
    constexpr double c3 = 3.0 / 8.0;
    constexpr double c4 = 12.0 / 13.0;
    constexpr double c5 = 1.0;
    constexpr double c6 = 1.0 / 2.0;

    constexpr double a21 = 1.0 / 4.0;
    constexpr double a31 = 3.0 / 32.0;
    constexpr double a32 = 9.0 / 32.0;
    constexpr double a41 = 1932.0 / 2197.0;
    constexpr double a42 = -7200.0 / 2197.0;
    constexpr double a43 = 7296.0 / 2197.0;
    constexpr double a51 = 439.0 / 216.0;
    constexpr double a52 = -8.0;
    constexpr double a53 = 3680.0 / 513.0;
    constexpr double a54 = -845.0 / 4104.0;
    constexpr double a61 = -8.0 / 27.0;
    constexpr double a62 = 2.0;
    constexpr double a63 = -3544.0 / 2565.0;
    constexpr double a64 = 1859.0 / 4104.0;
    constexpr double a65 = -11.0 / 40.0;

    constexpr double b1 = 16.0 / 135.0;
    constexpr double b2 = 0.0;
    constexpr double b3 = 6656.0 / 12825.0;
    constexpr double b4 = 28561.0 / 56430.0;
    constexpr double b5 = -9.0 / 50.0;
    constexpr double b6 = 2.0 / 55.0;

    constexpr double bs1 = 25.0 / 216.0;
    constexpr double bs2 = 0.0;
    constexpr double bs3 = 1408.0 / 2565.0;
    constexpr double bs4 = 2197.0 / 4104.0;
    constexpr double bs5 = -1.0 / 5.0;
    constexpr double bs6 = 0.0;

    double hmax = 0.9 * (xf - x); // maximum step size
    double h = hmax;

    constexpr int max_iter = 100;
    int iter = 0;

    while (x <= xf - h) {
        // clang-format off
        auto k1 = f(x + c1 * h, y);
        auto k2 = f(x + c2 * h, y + h * (a21 * k1));
        auto k3 = f(x + c3 * h, y + h * (a31 * k1 + a32 * k2));
        auto k4 = f(x + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
        auto k5 = f(x + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
        auto k6 = f(x + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

        auto err_vec = (b1 - bs1) * k1 + (b2 - bs2) * k2 + (b3 - bs3) * k3 + (b4 - bs4) * k4 + (b5 - bs5) * k5 + (b6 - bs6) * k6;
        err_vec *= h;
        double trunc_err = Scilib::Linalg::norm2(err_vec.view());
        if (trunc_err <= tol) {
            x += h;
            y += h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
        }
        // clang-format on
        h = 0.9 * h * std::pow(tol / trunc_err, 0.2);
        if (h < hmin) {
            h = hmin;
        }
        if (h > hmax) {
            h = hmax;
        }
        if (x + h > xf) {
            h = xf - x;
        }
        ++iter;
    }
    if (iter > max_iter) {
        std::runtime_error("rk45 failed to converge");
    }
}
