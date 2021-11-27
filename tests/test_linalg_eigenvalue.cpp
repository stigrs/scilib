// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalgEigenvalue, TestEigs)
{
    // Numpy:
    // clang-format off
    std::vector<double> eval = {
        3.28792877e-06, 3.05898040e-04, 1.14074916e-02, 2.08534219e-01, 
        1.56705069e+00
    };
    std::vector<double> evec_data = {
        -0.0062,  0.0472,  0.2142, -0.6019, 0.7679,
         0.1167, -0.4327, -0.7241,  0.2759, 0.4458,
        -0.5062,  0.6674, -0.1205,  0.4249, 0.3216,
         0.7672,  0.2330,  0.3096,  0.4439, 0.2534,
        -0.3762, -0.5576,  0.5652,  0.4290, 0.2098
    };
    std::vector<double> a_data = {
        1.0, 0.5, 1. / 3., 1. / 4., 1. / 5,
        0.5, 1. / 3., 1. / 4., 1. / 5., 1. / 6.,
        1. / 3., 1. / 4., 1. / 5., 1. / 6., 1. / 7.,
        1. / 4., 1. / 5., 1. / 6., 1. / 7., 1. / 8.,
        1. / 5., 1. / 6., 1. / 7., 1. / 8., 1. / 9.
    };
    // clang-format on
    Sci::Matrix<double> evec(evec_data, 5, 5);

    Sci::Matrix<double> a(a_data, 5, 5);
    Sci::Vector<double> w(5);
    Sci::Linalg::eigs(a.view(), w.view());

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::abs(w(i) - eval[i]) < 1.0e-8);
    }
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_TRUE(std::abs(a(i, j) - evec(i, j)) < 1.0e-4);
        }
    }
}
