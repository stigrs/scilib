// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/integrate.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestIntegrate, TestTrapz)
{
    std::vector<double> y_data = {3.2, 2.7, 2.9, 3.5, 4.1, 5.2};
    Scilib::Vector<double> y(y_data, y_data.size());

    double xlo = 2.1;
    double xup = 3.6;
    double ft = Scilib::Integrate::trapz(xlo, xup, y.view());

    EXPECT_NEAR(ft, 5.22, 1.0e-8);
}
