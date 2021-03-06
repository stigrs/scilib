// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <iostream>
#include <scilib/mdarray.h>
#include <sstream>
#include <string>

int main()
{
    using namespace Sci;

    std::string buf = "3 x 4\n \
        { 1  2  3  4\n \
          5  6  7  8\n \
          9 10 11 12 }";

    Matrix<int, layout_left> m_col_major;
    Matrix<int, layout_right> m_row_major;

    std::stringstream ss(buf);
    ss >> m_col_major;
    std::cout << "Column-major:\n" << m_col_major << "\n\n";

    ss.str(buf);
    ss >> m_row_major;
    std::cout << "Row-major:\n" << m_row_major << "\n\n";
}
