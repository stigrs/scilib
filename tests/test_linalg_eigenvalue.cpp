// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(TestLinalg, TestEigs)
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

TEST(TestLinalg, TestEig)
{
    // Numpy:
    std::vector<double> eval_re = {-3.17360337, -3.17360337, 2.84219813,
                                   7.50500862};

    std::vector<double> eval_im = {1.12844169, -1.12844169, 0.0, 0.0};

    // clang-format off
    std::vector<double> evec_re_data = {
        -0.16889612, -0.16889612, -0.19514446, 0.70845976,
         0.61501958,  0.61501958,  0.08601687, 0.46590401,
        -0.19838031, -0.19838031, -0.58764782, 0.52110625,
        -0.72497646, -0.72497646,  0.78050610, 0.09729590
    };
    std::vector<double> evec_im_data = {
        -0.11229493,  0.11229493, 0.0, 0.0,
        -0.03942734,  0.03942734, 0.0, 0.0,
         0.11880544, -0.11880544, 0.0, 0.0,
         0.0,         0.0,        0.0, 0.0
    };
    std::vector<double> a_data = {
         1.0, 5.0,  4.0,  2.0,
        -2.0, 3.0,  6.0,  4.0,
         5.0, 1.0,  0.0, -1.0,
         2.0, 3.0, -4.0,  0.0
    };
    // clang-format on

    Sci::Matrix<double> evec_re(evec_re_data, 4, 4);
    Sci::Matrix<double> evec_im(evec_im_data, 4, 4);
    Sci::Matrix<double> a(a_data, 4, 4);

    Sci::Vector<std::complex<double>> eval(4);
    Sci::Matrix<std::complex<double>> evec(4, 4);

    Sci::Linalg::eig(a.view(), evec.view(), eval.view());

    for (std::size_t i = 0; i < eval.size(); ++i) {
        EXPECT_TRUE(std::abs(eval(i).real() - eval_re[i]) < 5.0e-8);
        EXPECT_TRUE(std::abs(eval(i).imag() - eval_im[i]) < 5.0e-8);
    }

    for (std::size_t i = 0; i < evec.rows(); ++i) {
        for (std::size_t j = 0; j < evec.cols(); ++j) {
            EXPECT_TRUE(std::abs(evec(i, j).real() - evec_re(i, j)) < 5.0e-9);
            EXPECT_TRUE(std::abs(evec(i, j).imag() - evec_im(i, j)) < 5.0e-9);
        }
    }
}
