// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_RK45_H
#define SCILIB_INTEGRATE_RK45_H

#include <functional>
#include <scilib/mdarray.h>
#include <string>

namespace Scilib {
namespace Integrate {
namespace __Detail {

// Explicit Runge-Kutta-Fehlberg method of order 5(4).
void rk45(std::function<
              Scilib::Vector<double>(double, const Scilib::Vector<double>&)> f,
          double& x,
          double xf,
          Scilib::Vector<double>& y,
          double tol = 1.0e-6,
          double hmin = 1.0e-9);

// Explicit Dormand-Prince method of order 5.
void dopri5(
    std::function<Scilib::Vector<double>(double, const Scilib::Vector<double>&)>
        f,
    double& x,
    double xf,
    Scilib::Vector<double>& y,
    double tol = 1.0e-6,
    double hmin = 1.0e-9);

} // namespace __Detail

// Solve an inital value problem for a system of ODEs.
//
// dy/dx = f(x, y)
// y(x0) = y0
//
inline void solve_ivp(
    std::function<Scilib::Vector<double>(double, const Scilib::Vector<double>&)>
        f,
    double& x,
    double xf,
    Scilib::Vector<double>& y,
    const std::string& method = "DOPRI5",
    double tol = 1.0e-6,
    double hmin = 1.0e-9)
{
    if (method == "RK45" || method == "rk45") {
        __Detail::rk45(f, x, xf, y, tol, hmin);
    }
    else {
        __Detail::dopri5(f, x, xf, y, tol, hmin);
    }
}

} // namespace Integrate
} // namespace Scilib

#endif // SCILIB_INTEGRATE_RK45_H
