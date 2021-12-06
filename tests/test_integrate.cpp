// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/integrate.h>
#include <scilib/constants.h>
#include <gtest/gtest.h>
#include <vector>
#include <limits>

Scilib::Vector<double> lorentz(double /* t */, const Scilib::Vector<double>& y)
{
    const double sigma = 10.0;
    const double R = 28.0;
    const double b = 8.0 / 3.0;

    Scilib::Vector<double> ydot(3);

    ydot(0) = sigma * (y(1) - y(0));
    ydot(1) = R * y(0) - y(1) - y(0) * y(2);
    ydot(2) = -b * y(2) + y(0) * y(1);

    return ydot;
}

TEST(TestIntegrate, TestTrapz)
{
    std::vector<double> y_data = {3.2, 2.7, 2.9, 3.5, 4.1, 5.2};
    Scilib::Vector<double> y(y_data, y_data.size());

    double xlo = 2.1;
    double xup = 3.6;
    double ft = Scilib::Integrate::trapz(xlo, xup, y.view());

    EXPECT_NEAR(ft, 5.22, 1.0e-8);
}

TEST(TestIntegrate, TestQuad)
{
    using namespace Scilib;
    using namespace Scilib::Integrate;

    double a = 0.0;
    double b = Constants::pi;

    double res = quad<5>([](double x) { return std::sin(x); }, a, b);
    EXPECT_NEAR(res, 2.0, 5.0e-7);

    res = quad<8>([](double x) { return std::sin(x); }, a, b);
    EXPECT_NEAR(res, 2.0, 1.0e-14);

    double eps = std::numeric_limits<double>::epsilon();
    res = quad<16>([](double x) { return std::sin(x); }, a, b);
    EXPECT_NEAR(res, 2.0, eps);
}

TEST(TestIntegrate, TestDormandPrince)
{
    using namespace Scilib;
    using namespace Scilib::Integrate;

    // clang-format off
    std::vector<double> ans_data = { // result from Matlab
        12.420121076782189, 22.132678932307815, 11.996473826705991,
        19.500081683089384, 16.224736836476261, 45.258556702999961,
         6.613599319856808, -7.931580903108999, 37.735650643710017,
        -2.963989264539828, -8.250556890143775, 28.287476810924446,
        -6.217033890199554, -8.278471219613175, 25.168552598624345
    };
    // clang-format on
    Matrix<double> ans(ans_data, 5, 3);

    std::vector<double> y0 = {10.0, 1.0, 1.0};
    Vector<double> y(y0, 3);

    double t0 = 0.0;
    double tf = 0.1;

    for (int i = 0; i < 5; ++i) {
        solve_ivp(lorentz, t0, tf, y);
        for (std::size_t j = 0; j < y.extent(0); ++j) {
            EXPECT_NEAR(y(j), ans(i, j), 1.0e-5);
        }
        tf += 0.1;
    }
}
