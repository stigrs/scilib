// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/integrate.h>
#include <cmath>

double
Scilib::Integrate::trapz(double xlo, double xup, Scilib::Vector_view<double> y)
{
    using size_type = Scilib::Vector_view<double>::size_type;

    const double step = std::abs(xup - xlo) / (y.size() - 1);
    double ans = 0.0;

    for (size_type i = 1; i < y.size(); ++i) {
        ans += 0.5 * (y(i) + y(i - 1));
    }
    return ans *= step;
}
