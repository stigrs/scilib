// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#include <gtest/gtest.h>
#include <scilib/mdarray.h>
#include <utility>
#include <vector>


template <class Container, class Extents>
Sci::Array4D<int> make_array4d(const Container& ctr, const Extents& exts)
{
    return Sci::Array4D<int>(exts, ctr);
}

TEST(TestMDArray, TestArray4D)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 1;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; j < a.extent(1); ++j) {
            for (index_type k = 0; k < a.extent(2); ++k) {
                for (index_type l = 0; l < a.extent(3); ++l) {
                    EXPECT_EQ(a(i, j, k, l), 1);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DVector)
{
    using index_type = Sci::Array4D<int>::index_type;

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    Sci::Array4D<int> a(2, 2, 2, 2);
    Sci::Array4D<int> b(a.mapping(), data);

    int it = 1;
    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), it);
                    ++it;
                }
            }
        }
    }

    Sci::Array4D<int> c = make_array4d(data, a.extents());
    for (std::size_t r = 0; r < c.rank(); ++r) {
        EXPECT_EQ(c.extent(r), a.extent(r));
    }

    Sci::Array4D<int> d(a.mapping(), std::move(data));
    for (std::size_t r = 0; r < c.rank(); ++r) {
        EXPECT_EQ(d.extent(r), a.extent(r));
    }
}

TEST(TestMDArray, TestCopyArrayRowMajorColMajor)
{
    using extents_type = Sci::Array4D<int>::extents_type;
    using index_type = Sci::Array4D<int>::index_type;

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    Sci::Array4D<int> a(extents_type(2, 2, 2, 2), data);
    Sci::Array4D<int, Mdspan::layout_left> b(a.to_mdspan());

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; j < a.extent(1); ++j) {
            for (index_type k = 0; k < a.extent(2); ++k) {
                for (index_type l = 0; l < a.extent(3); ++l) {
                    EXPECT_EQ(a(i, j, k, l), b(i, j, k, l));
                }
            }
        }
    }
}

TEST(TestMDArray, TestCopyArrayColMajorRowMajor)
{
    using extents_type = Sci::Array4D<int, Mdspan::layout_left>::extents_type;
    using index_type = Sci::Array4D<int>::index_type;

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    Sci::Array4D<int, Mdspan::layout_left> a(extents_type(2, 2, 2, 2), data);
    Sci::Array4D<int> b(a.to_mdspan());

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), a(i, j, k, l));
                }
            }
        }
    }
}

TEST(TestMDArray, TestCopyArrayColMajorRowMajor2)
{
    using extents_type = Sci::Array4D<int, Mdspan::layout_left>::extents_type;
    using index_type = Sci::Array4D<int>::index_type;

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    Sci::Array4D<int, Mdspan::layout_left> a(extents_type(2, 2, 2, 2), data);
    Sci::Array4D<int> b(2, 2, 2, 2);

    Sci::copy(a.to_mdspan(), b.to_mdspan());

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), a(i, j, k, l));
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DMinusOperator)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 1;
    a = -a;

    for (index_type i = 0; i < a.extent(0); ++i) {
        for (index_type j = 0; j < a.extent(1); ++j) {
            for (index_type k = 0; k < a.extent(2); ++k) {
                for (index_type l = 0; l < a.extent(3); ++l) {
                    EXPECT_EQ(a(i, j, k, l), -1);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DAddScalar)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 1;
    auto b = a + 1;

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), 2);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DSubtractScalar)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 2;
    auto b = a - 1;

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), 1);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DMultiplyScalar)
{
    using index_type = Sci::Array4D<int>::index_type;

    Sci::Array4D<int> a(3, 4, 5, 6);
    a = 2;
    auto b = a * 2;

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), 4);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DDivideScalar)
{
    using index_type = Sci::Array4D<double>::index_type;

    Sci::Array4D<double> a(3, 4, 5, 6);
    a = 4.0;
    auto b = a / 2.0;

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), 2.0);
                }
            }
        }
    }
}

TEST(TestMDArray, TestArray4DApply)
{
    using index_type = Sci::Array4D<double>::index_type;

    Sci::Array4D<double> a(3, 4, 5, 6);
    Sci::Array4D<double> b(3, 4, 5, 6);

    a = 4.0;
    b = 1.0;

    Sci::apply(a.to_mdspan(), b.to_mdspan(), [&](double& x, double& y) { y += x + 1.0; });

    for (index_type i = 0; i < b.extent(0); ++i) {
        for (index_type j = 0; j < b.extent(1); ++j) {
            for (index_type k = 0; k < b.extent(2); ++k) {
                for (index_type l = 0; l < b.extent(3); ++l) {
                    EXPECT_EQ(b(i, j, k, l), 6.0);
                }
            }
        }
    }
}
