// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>
#include <vector>

TEST(TestLinalg, TestEigs)
{
    // Intel MKL example:
    // clang-format off
    std::vector<double> eval = {
        -17.44, -11.96, 6.72, 14.25, 19.84
    };
    std::vector<double> evec_data = {
        -0.26,  0.31, -0.74,  0.33,  0.42,
        -0.17, -0.39, -0.38, -0.80,  0.16,
        -0.89,  0.04,  0.09,  0.03, -0.45,
        -0.29, -0.59,  0.34,  0.31,  0.60,
        -0.19,  0.63,  0.44, -0.38,  0.48
    };
    std::vector<double> a_data = {
        6.39,  0.13, -8.23,  5.71, -3.18,
        0.00,  8.37, -4.46, -6.10,  7.21,
        0.00,  0.00, -9.58, -9.25, -7.42,
        0.00,  0.00,  0.00,  3.72,  8.54,
        0.00,  0.00,  0.00,  0.00,  2.51
    };
    // clang-format on
    Sci::Matrix<double> evec(evec_data, 5, 5);

    Sci::Matrix<double> a(a_data, 5, 5);
    Sci::Vector<double> w(5);
    Sci::Linalg::eigs(a.view(), w.view());

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(w(i), eval[i], 1.0e-2);
    }

#ifdef USE_MKL
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(a(i, j), evec(i, j), 1.0e-2);
        }
    }
#else
    for (int i = 0; i < 5; ++i) {
        for (int j = i; j < 5; ++j) {
            EXPECT_NEAR(a(i, j), evec(i, j), 1.0e-2);
        }
    }
#endif
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
        EXPECT_NEAR(eval(i).real(), eval_re[i], 5.0e-8);
        EXPECT_NEAR(eval(i).imag(), eval_im[i], 5.0e-8);
    }

    for (std::size_t i = 0; i < evec.extent(0); ++i) {
        for (std::size_t j = 0; j < evec.extent(1); ++j) {
            EXPECT_NEAR(evec(i, j).real(), evec_re(i, j), 5.0e-9);
            EXPECT_NEAR(evec(i, j).imag(), evec_im(i, j), 5.0e-9);
        }
    }
}

