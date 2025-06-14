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

TEST(TestLinalg, TestMatrixPower)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // Example from Numpy:
    // clang-format off
    const std::vector<double> ans1_data = {
        0.0, -1.0, 
        1.0,  0.0};
    const std::vector<double> ans2_data = {
        1.0, 0.0, 
        0.0, 1.0};
    const std::vector<double> ans3_data = {
         0.0, 1.0, 
        -1.0, 0.0};
    const std::vector<double> ans4_data = {
        -1.0,  0.0, 
         0.0, -1.0};
    const std::vector<double> M_data = {
         0.0, 1.0, 
        -1.0, 0.0};
    // clang-format on
    using extents_type = typename Matrix<double>::extents_type;

    Matrix<double> ans1(extents_type(2, 2), ans1_data);
    Matrix<double> ans2(extents_type(2, 2), ans2_data);
    Matrix<double> ans3(extents_type(2, 2), ans3_data);
    Matrix<double> ans4(extents_type(2, 2), ans4_data);

    Matrix<double> M(extents_type(2, 2), M_data);

    auto res = matrix_power(M, 0);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans2(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 1);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), M(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 2);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans4(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 3);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans1(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, -3);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans3(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 6);
    for (Sci::index i = 0; i < res.extent(0); ++i) {
        for (Sci::index j = 0; j < res.extent(1); ++j) {
            EXPECT_NEAR(res(i, j), ans4(i, j), 1.0e-6);
        }
    }
}

TEST(TestLinalg, TestMatrixPowerColMajor)
{
    using namespace Sci;
    using namespace Sci::Linalg;

    // Example from Numpy:
    // clang-format off
    const std::vector<double> ans1_data = {
         0.0, 1.0, 
        -1.0, 0.0};
    const std::vector<double> ans2_data = {
        1.0, 0.0, 
        0.0, 1.0};
    const std::vector<double> ans3_data = {
        0.0, -1.0, 
        1.0,  0.0};
    const std::vector<double> ans4_data = {
        -1.0,  0.0, 
         0.0, -1.0};
    const std::vector<double> M_data = {
        0.0, -1.0, 
        1.0,  0.0};
    // clang-format on
    using extents_type = typename Matrix<double, Mdspan::layout_left>::extents_type;

    Matrix<double, Mdspan::layout_left> ans1(extents_type(2, 2), ans1_data);
    Matrix<double, Mdspan::layout_left> ans2(extents_type(2, 2), ans2_data);
    Matrix<double, Mdspan::layout_left> ans3(extents_type(2, 2), ans3_data);
    Matrix<double, Mdspan::layout_left> ans4(extents_type(2, 2), ans4_data);

    Matrix<double, Mdspan::layout_left> M(extents_type(2, 2), M_data);

    auto res = matrix_power(M, 0);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans2(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 1);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), M(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 2);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans4(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 3);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans1(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, -3);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans3(i, j), 1.0e-6);
        }
    }
    res = matrix_power(M, 6);
    for (Sci::index j = 0; j < res.extent(1); ++j) {
        for (Sci::index i = 0; i < res.extent(0); ++i) {
            EXPECT_NEAR(res(i, j), ans4(i, j), 1.0e-6);
        }
    }
}
