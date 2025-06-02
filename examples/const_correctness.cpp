#include <iostream>
#include <scilib/mdarray.h>

void beast(const Sci::Matrix<int>& a)
{
    using namespace Sci;

    Matrix<int> b = slice(a, Mdspan::full_extent, Mdspan::full_extent);
    b(1, 1) = 666;
}

int main()
{
    using namespace Sci;

    Matrix<int> a(5, 5);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            a(i, j) = i + j;
        }
    }
    std::cout << "Before:\n" << a << '\n';
    beast(a);
    std::cout << "After:\n" << a << '\n';
}
