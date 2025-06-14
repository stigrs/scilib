// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <cmath>
#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>
#include <vector>

TEST(TestLinalg, TestEighReal)
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
    Sci::Matrix<double> evec(Mdspan::dextents<Sci::index, 2>(5, 5), evec_data);

    Sci::Matrix<double> a(Mdspan::dextents<Sci::index, 2>(5, 5), a_data);
    Sci::Vector<double> w(5);
    Sci::Linalg::eigh(a, w);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(w(i), eval[i], 1.0e-6);
    }

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(std::abs(a(i, j)), std::abs(evec(i, j)), 1.0e-6);
        }
    }
}

TEST(TestLinalg, TestEighComplex)
{
    // Intel MKL example:
    // clang-format off
    Sci::Matrix<std::complex<double>> a = {
        {{-2.16,  0.00}, { 0.00,  0.00}, { 0.00,  0.00}, { 0.00,  0.00}},
        {{-0.16,  4.86}, { 7.45,  0.00}, { 0.00,  0.00}, { 0.00,  0.00}},
        {{-7.23,  9.38}, { 4.39, -6.29}, {-9.03,  0.00}, { 0.00,  0.00}},
        {{-0.04, -6.86}, {-8.11,  4.41}, {-6.89,  7.66}, { 7.76,  0.00}}
    };
    Sci::Matrix<std::complex<double>> evec = {
        {{ 0.68, 0.00}, { 0.38,  0.00}},
        {{ 0.03, 0.18}, { 0.54, -0.57}},
        {{-0.03, 0.21}, {-0.40,  0.04}},
        {{ 0.20, 0.64}, {-0.14, -0.26}}
    };
    Sci::Vector<double> eval = {-4.18, 3.57};
    // clang-format on

    Sci::Vector<double> w(a.extent(1));
    Sci::Linalg::eigh(a, w, 'L');

    for (int i = 1; i < 3; ++i) {
        EXPECT_NEAR(w(i), eval(i - 1), 5.0e-3);
    }
    for (int i = 0; i < a.extent(0); ++i) {
        for (int j = 1; j < 3; ++j) {
            EXPECT_NEAR(std::abs(a(i, j)), std::abs(evec(i, j - 1)), 5.0e-3);
        }
    }
}

TEST(TestLinalg, TestEig)
{
    // Numpy:
    std::vector<double> eval_re = {-3.17360337, -3.17360337, 2.84219813, 7.50500862};

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
    using extents_type = typename Sci::Matrix<double>::extents_type;

    Sci::Matrix<double> evec_re(extents_type(4, 4), evec_re_data);
    Sci::Matrix<double> evec_im(extents_type(4, 4), evec_im_data);
    Sci::Matrix<double> a(extents_type(4, 4), a_data);

    Sci::Vector<std::complex<double>> eval(4);
    Sci::Matrix<std::complex<double>> evec(4, 4);

    Sci::Linalg::eig(a, evec, eval);

    for (std::size_t i = 0; i < eval.size(); ++i) {
        EXPECT_NEAR(eval(i).real(), eval_re[i], 5.0e-8);
        EXPECT_NEAR(eval(i).imag(), eval_im[i], 5.0e-8);
    }

    for (Sci::index i = 0; i < evec.extent(0); ++i) {
        for (Sci::index j = 0; j < evec.extent(1); ++j) {
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
    std::vector<double> eval_re = {-3.17360337, -3.17360337, 2.84219813, 7.50500862};

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
    using extents_type = typename Matrix<double, Mdspan::layout_left>::extents_type;

    Matrix<double, Mdspan::layout_left> evec_re(extents_type(4, 4), evec_re_data);
    Matrix<double, Mdspan::layout_left> evec_im(extents_type(4, 4), evec_im_data);
    Matrix<double, Mdspan::layout_left> a(extents_type(4, 4), a_data);

    Vector<std::complex<double>, Mdspan::layout_left> eval(4);
    Matrix<std::complex<double>, Mdspan::layout_left> evec(4, 4);

    eig(a, evec, eval);

    for (std::size_t i = 0; i < eval.size(); ++i) {
        EXPECT_NEAR(eval(i).real(), eval_re[i], 5.0e-8);
        EXPECT_NEAR(eval(i).imag(), eval_im[i], 5.0e-8);
    }

    for (Sci::index j = 0; j < evec.extent(1); ++j) {
        for (Sci::index i = 0; i < evec.extent(0); ++i) {
            EXPECT_NEAR(evec(i, j).real(), evec_re(i, j), 5.0e-9);
            EXPECT_NEAR(evec(i, j).imag(), evec_im(i, j), 5.0e-9);
        }
    }
}
