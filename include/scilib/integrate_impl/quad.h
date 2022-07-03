// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_INTEGRATE_QUAD_H
#define SCILIB_INTEGRATE_QUAD_H

#include <array>
#include <functional>

namespace Sci {
namespace Integrate {

// Return tabulated roots and weights for a Gauss-Legendre quadrature
// of order n.
template <int N = 8>
inline void gauss_legendre(std::array<double, N>& roots,
                           std::array<double, N>& weights,
                           double a = -1.0,
                           double b = 1.0)
{
    static_assert(N == 5 || N == 8 || N == 16, "bad order for Gauss-Legendre quadrature");

    // 5-point:
    std::array<double, 5> x5{
        0.00000000000000000,
        -0.5384693101056831,
        0.5384693101056831,
        -0.9061798459386640,
        0.9061798459386640,
    };
    std::array<double, 5> w5{
        0.5688888888888889,
        0.4786286704993665,
        0.4786286704993665,
        0.2369268850561891,
        0.2369268850561891,
    };
    std::array<double, 8> x8{-0.1834346424956498, 0.1834346424956498, -0.5255324099163290,
                             0.5255324099163290, -0.7966664774136267, 0.7966664774136267,
                             -0.9602898564975363, 0.9602898564975363};
    std::array<double, 8> w8{0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                             0.3137066458778873, 0.2223810344533745, 0.2223810344533745,
                             0.1012285362903763, 0.1012285362903763};

    std::array<double, 16> x16{
        -0.0950125098376374, 0.0950125098376374, -0.2816035507792589, 0.2816035507792589,
        -0.4580167776572274, 0.4580167776572274, -0.6178762444026438, 0.6178762444026438,
        -0.7554044083550030, 0.7554044083550030, -0.8656312023878318, 0.8656312023878318,
        -0.9445750230732326, 0.9445750230732326, -0.9894009349916499, 0.9894009349916499};
    std::array<double, 16> w16{
        0.1894506104550685, 0.1894506104550685, 0.1826034150449236, 0.1826034150449236,
        0.1691565193950025, 0.1691565193950025, 0.1495959888165767, 0.1495959888165767,
        0.1246289712555339, 0.1246289712555339, 0.0951585116824928, 0.0951585116824928,
        0.0622535239386479, 0.0622535239386479, 0.0271524594117541, 0.0271524594117541};

    switch (N) {
    case 16:
        std::copy(x16.begin(), x16.end(), roots.begin());
        std::copy(w16.begin(), w16.end(), weights.begin());
        break;
    case 5:
        std::copy(x5.begin(), x5.end(), roots.begin());
        std::copy(w5.begin(), w5.end(), weights.begin());
        break;
    case 8:
    default:
        std::copy(x8.begin(), x8.end(), roots.begin());
        std::copy(w8.begin(), w8.end(), weights.begin());
    }

    // Change of interval:
    for (int i = 0; i < N; ++i) {
        roots[i] = 0.5 * (b - a) * roots[i] + 0.5 * (a + b);
        weights[i] *= 0.5 * (b - a);
    }
}

// Integrate function from a to b using a Gauss-Legendre quadrature
// of order N.
template <int N = 8>
inline double quad(std::function<double(double)> f, double a, double b)
{
    static_assert(N == 5 || N == 8 || N == 16, "bad order for Gauss-Legendre quadrature");

    std::array<double, N> x;
    std::array<double, N> w;

    gauss_legendre<N>(x, w, a, b);

    double res = 0.0;
    for (int i = 0; i < N; ++i) {
        res += w[i] * f(x[i]);
    }
    return res;
}

} // namespace Integrate
} // namespace Sci

#endif // SCILIB_INTEGRATE_QUAD_H
