#include <iostream>
#include <scilib/mdarray.h>
#include <vector>

using namespace Sci;

int main()
{
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<int> mat(Kokkos::dextents<Sci::index, 2>(3, 3), data);

    std::cout << "Before:\n" << mat << '\n';

    auto sub1 = slice(mat, seq(1, 3), seq(1, 3));
    auto sub2 = slice(mat, seq(0, 2), seq(0, 2));
    copy(sub2, sub1);
    std::cout << "After:\n" << mat << '\n';

    mat = Matrix<int>(Kokkos::dextents<Sci::index, 2>(3, 3), data);
    std::cout << "Before:\n" << mat << '\n';

    Matrix<int> bv = slice(mat, seq(0, 2), seq(0, 2));
    sub1 = slice(mat, seq(1, 3), seq(1, 3));
    copy(bv.to_mdspan(), sub1);
    std::cout << "After:\n" << mat << '\n';
}