// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <scilib/mdarray.h>
#include <scilib/linalg.h>
#include <gtest/gtest.h>

TEST(TestLinalg, TestMatrixMatrixProductInt)
{
    // clang-format off
    std::vector<int> aa = {1, 2, 3, 
                           4, 5, 6};
    std::vector<int> bb = { 7, 8, 
                            9, 10, 
                           11, 12};
    std::vector<int> rr = { 58,  64, 
                           139, 154};
    // clang-format on

    Scilib::Matrix<int> ma(aa, 2, 3);
    Scilib::Matrix<int> mb(bb, 3, 2);
    Scilib::Matrix<int> ans(rr, 2, 2);
    Scilib::Matrix<int> res = ma * mb;

    EXPECT_EQ(ans, res);
}

TEST(TestLinalg, TestMatrixMatrixProductDouble)
{
    // clang-format off
    std::vector<double> aa = {1, 2, 3, 
                              4, 5, 6};
    std::vector<double> bb = { 7, 8, 
                               9, 10, 
                              11, 12};
    std::vector<double> rr = { 58,  64, 
                              139, 154};
    // clang-format on

    Scilib::Matrix<double> ma(aa, 2, 3);
    Scilib::Matrix<double> mb(bb, 3, 2);
    Scilib::Matrix<double> ans(rr, 2, 2);
    Scilib::Matrix<double> res = ma * mb;

    EXPECT_EQ(ans, res);
}

TEST(TestLinalg, TestMatrixMatrixProductComplex)
{
    // clang-format off
    std::vector<std::complex<double>> aa = {{1.0, 2.0}, {3.0, 4.0}, 
                                            {5.0, 6.0}, {7.0, 8.0}};
    std::vector<std::complex<double>> bb = {{1.0, -2.0}, {3.0, -4.0}, 
                                            {5.0, -6.0}, {7.0, -8.0}};
    std::vector<std::complex<double>> rr = {{44.0, 2.0}, {64.0, 6.0}, 
                                            {100.0, -6.0}, {152.0, -2.0}};
    // clang-format on

    Scilib::Matrix<std::complex<double>> ma(aa, 2, 2);
    Scilib::Matrix<std::complex<double>> mb(bb, 2, 2);
    Scilib::Matrix<std::complex<double>> ans(rr, 2, 2);
    Scilib::Matrix<std::complex<double>> res = ma * mb;

    EXPECT_EQ(ans, res);
}
