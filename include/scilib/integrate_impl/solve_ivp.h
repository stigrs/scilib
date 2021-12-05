// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_SOLVE_IVP_H
#define SCILIB_INTEGRATE_SOLVE_IVP_H

#include <scilib/mdarray.h>
#include <functional>
#include <string>

namespace Scilib {
namespace Integrate {
namespace __Detail {

// Embedded Runge-Kutta-Fehlberg method of order 4(5).
void runge_kutta_fehlberg(
    std::function<Scilib::Vector<double>(double, const Scilib::Vector<double>&)>
        f,
    double& x,
    double xf,
    Scilib::Vector<double>& y,
    double tol,
    double hmin);

// Embedded Dormand-Prince method of order 4(5).
void dormand_prince(
    std::function<Scilib::Vector<double>(double, const Scilib::Vector<double>&)>
        f,
    double& x,
    double xf,
    Scilib::Vector<double>& y,
    double tol,
    double hmin);

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
    const std::string& method = "Dormand-Prince",
    double tol = 1.0e-6,
    double hmin = 1.0e-9)
{
    if (method == "Runge-Kutta-Fehlberg") {
        __Detail::runge_kutta_fehlberg(f, x, xf, y, tol, hmin);
    }
    else {
        __Detail::dormand_prince(f, x, xf, y, tol, hmin);
    }
}

} // namespace Integrate
} // namespace Scilib

#endif // SCILIB_INTEGRATE_SOLVE_IVP_H
