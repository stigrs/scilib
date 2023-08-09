// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <scilib/constants.h>
#include <scilib/integrate.h>
#include <scilib/mdarray.h>
#include <vector>

Sci::Vector<double> lorentz(double, const Sci::Vector<double>& y)
{
    const double sigma = 10.0;
    const double R = 28.0;
    const double b = 8.0 / 3.0;

    Sci::Vector<double> ydot(3);

    ydot(0) = sigma * (y(1) - y(0));
    ydot(1) = R * y(0) - y(1) - y(0) * y(2);
    ydot(2) = -b * y(2) + y(0) * y(1);

    return ydot;
}

Sci::StaticVector<double, 3> fsys_stiff(double, const Sci::StaticVector<double, 3>& y)
{
    Sci::StaticVector<double, 3> ydot(3);

    ydot(0) = -0.04 * y(0) + 1.0e4 * y(1) * y(2);
    ydot(2) = 3.0e7 * y(1) * y(1);
    ydot(1) = -ydot(0) - ydot(2);

    return ydot;
}

TEST(TestIntegrate, TestTrapz)
{
    Sci::Vector<double> y = {3.2, 2.7, 2.9, 3.5, 4.1, 5.2};

    double xlo = 2.1;
    double xup = 3.6;
    double ft = Sci::Integrate::trapz(xlo, xup, y);

    EXPECT_NEAR(ft, 5.22, 1.0e-8);
}

TEST(TestIntegrate, TestQuad)
{
    using namespace Sci;
    using namespace Sci::Integrate;

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
    using namespace Sci;
    using namespace Sci::Integrate;

    // clang-format off
    std::vector<double> ans_data = { // from Scipy using DOP853 with atol=1.0e-y, rtol=1.0e-7
        12.4201224814, 22.1326749017,  11.9964784533,
        19.5000801046, 16.2247395620,  45.2585473265,
         6.6136027295, -7.9315751250,  37.7356530183,
        -2.9639869312, -8.2505557857,  28.2874739170,
        -6.2170338617, -8.2784722478,  25.1685488579,
        -7.9977242247, -9.6524544581,  24.9327594749,
        -9.4444100181, -10.5068331831, 27.0135732175,
        -9.7570772218, -9.2388972510,  29.2558718461,
        -8.6291856308, -7.1555905793,  29.0007439931,
        -7.3535355376, -6.4755890802,  26.8363589291
    };
    // clang-format on
    using extents_type = typename Matrix<double>::extents_type;
    Matrix<double> ans(extents_type(5, 3), ans_data);

    std::vector<double> y0 = {10.0, 1.0, 1.0};
    Vector<double> y(stdex::dextents<Sci::index, 1>(y0.size()), y0);

    double t0 = 0.0;
    double tf = 0.1;

    for (int i = 0; i < 10; ++i) {
        solve_ivp(lorentz, t0, tf, y, 1.0e-7, 1.0e-7);
        for (Sci::index j = 0; j < y.extent(0); ++j) {
            EXPECT_NEAR(y(j), ans(i, j), 1.6e-6);
        }
        tf += 0.1;
    }
}

TEST(TestIntegrate, TestStiff)
{
    using namespace Sci;
    using namespace Sci::Integrate;

    // clang-format off
    std::vector<double> ans_data = { // result from Lsoda
        9.851712e-01, 3.386380e-05, 1.479493e-02,
        9.055333e-01, 2.240655e-05, 9.444430e-02,
        7.158403e-01, 9.186334e-06, 2.841505e-01,
    };
    // clang-format on
    using extents_type_mat = typename Matrix<double>::extents_type;
    using extents_type_vec = typename StaticVector<double, 3>::extents_type;

    Matrix<double> ans(extents_type_mat(3, 3), ans_data);

    std::array<double, 3> y0 = {1.0, 0.0, 0.0};
    StaticVector<double, 3> y(extents_type_vec(3), y0);

    double t0 = 0.0;
    double tf = 0.4;

    int i = 0;
    for (int it = 0; it < 100; ++it) {
        solve_ivp(fsys_stiff, t0, tf, y, 1.0e-7, 1.0e-7);
        if (it == 0 || it == 9 || it == 99) {
            for (Sci::index j = 0; j < y.extent(0); ++j) {
                EXPECT_NEAR(y(j), ans(i, j), 1.5e-5);
            }
            ++i;
        }
        tf += 0.4;
    }
}
