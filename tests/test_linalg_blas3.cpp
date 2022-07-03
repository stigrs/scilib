// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

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

    Sci::Matrix<int> ma(aa, 2, 3);
    Sci::Matrix<int> mb(bb, 3, 2);
    Sci::Matrix<int> ans(rr, 2, 2);
    Sci::Matrix<int> res = ma * mb;

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

    Sci::Matrix<double> ma(aa, 2, 3);
    Sci::Matrix<double> mb(bb, 3, 2);
    Sci::Matrix<double> ans(rr, 2, 2);
    Sci::Matrix<double> res(2, 2);

    Sci::Linalg::matrix_product(ma, mb, res);

    EXPECT_EQ(ans, res);
}

TEST(TestLinalg, TestMatrixMatrixProductDoubleColMajor)
{
    // clang-format off
    std::vector<double> aa = {1, 4, 
                              2, 5,
                              3, 6};
    std::vector<double> bb = { 7, 9, 11,
                               8, 10, 12};
    std::vector<double> rr = { 58,  139, 
                               64, 154};
    // clang-format on

    Sci::Matrix<double, stdex::layout_left> ma(aa, 2, 3);
    Sci::Matrix<double, stdex::layout_left> mb(bb, 3, 2);
    Sci::Matrix<double, stdex::layout_left> ans(rr, 2, 2);
    Sci::Matrix<double, stdex::layout_left> res(2, 2);

    Sci::Linalg::matrix_product(ma, mb, res);

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

    Sci::Matrix<std::complex<double>> ma(aa, 2, 2);
    Sci::Matrix<std::complex<double>> mb(bb, 2, 2);
    Sci::Matrix<std::complex<double>> ans(rr, 2, 2);
    Sci::Matrix<std::complex<double>> res(2, 2);

    Sci::Linalg::matrix_product(ma, mb, res);

    EXPECT_EQ(ans, res);
}
