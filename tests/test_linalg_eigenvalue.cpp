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
    // Intel MKL example:
    // clang-format off
    std::vector<double> eval = {
         0.4330218, 2.14494666, 3.36808674, 4.27915302, 6.93479178
    };
    std::vector<double> evec_data = {
       -0.9796240, -0.0146276, -0.0817668, -0.0124140,  0.182435,
        0.0109926,  0.0209833, -0.9338050,  0.0257257, -0.356068,
        0.0444859, -0.6927900, -0.0735199, -0.7086760,  0.102155,
       -0.1810480,  0.1934840,  0.3129500, -0.3541350, -0.840498,
        0.0738794,  0.6942270, -0.1340860, -0.6095500,  0.350800 
    };
    std::vector<double> a_data = {
        0.67, -0.20,  0.19, -1.06,  0.46,
       -0.20,  3.82, -0.13,  1.06, -0.48,
        0.19, -0.13,  3.27,  0.11,  1.10,
       -1.06,  1.06,  0.11,  5.86, -0.98,
        0.46, -0.48,  1.10, -0.98,  3.54
    };
    // clang-format on
    Sci::Matrix<double> evec(evec_data, 5, 5);

    Sci::Matrix<double> a(a_data, 5, 5);
    Sci::Vector<double> w(5);
    Sci::Linalg::eigs(a, w);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(w(i), eval[i], 1.0e-6);
    }

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(std::abs(a(i, j)), std::abs(evec(i, j)), 1.0e-6);
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

    Sci::Linalg::eig(a, evec, eval);

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

TEST(TestLinalg, TestEigColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // Numpy:
    std::vector<double> eval_re = {-3.17360337, -3.17360337, 2.84219813,
                                   7.50500862};

    std::vector<double> eval_im = {1.12844169, -1.12844169, 0.0, 0.0};

    // clang-format off
    std::vector<double> evec_re_data = {
        -0.16889612, 0.61501958, -0.19838031, -0.72497646, 
        -0.16889612, 0.61501958, -0.19838031, -0.72497646, 
        -0.19514446, 0.08601687, -0.58764782,  0.78050610, 
         0.70845976, 0.46590401,  0.52110625,  0.09729590
    };
    std::vector<double> evec_im_data = {
        -0.11229493, -0.03942734,  0.11880544, 0.0,       
         0.11229493,  0.03942734, -0.11880544, 0.0,        
         0.0,         0.0,         0.0,        0.0,
         0.0,         0.0,         0.0,        0.0
    };
    std::vector<double> a_data = {
         1.0, -2.0,  5.0,  2.0, 
         5.0,  3.0,  1.0,  3.0,
         4.0,  6.0,  0.0, -4.0,
         2.0,  4.0, -1.0,  0.0
    };
    // clang-format on

    Matrix<double, stdex::layout_left> evec_re(evec_re_data, 4, 4);
    Matrix<double, stdex::layout_left> evec_im(evec_im_data, 4, 4);
    Matrix<double, stdex::layout_left> a(a_data, 4, 4);

    Vector<std::complex<double>, stdex::layout_left> eval(4);
    Matrix<std::complex<double>, stdex::layout_left> evec(4, 4);

    eig(a, evec, eval);

    for (std::size_t i = 0; i < eval.size(); ++i) {
        EXPECT_NEAR(eval(i).real(), eval_re[i], 5.0e-8);
        EXPECT_NEAR(eval(i).imag(), eval_im[i], 5.0e-8);
    }

    for (std::size_t j = 0; j < evec.extent(1); ++j) {
        for (std::size_t i = 0; i < evec.extent(0); ++i) {
            EXPECT_NEAR(evec(i, j).real(), evec_re(i, j), 5.0e-9);
            EXPECT_NEAR(evec(i, j).imag(), evec_im(i, j), 5.0e-9);
        }
    }
}
