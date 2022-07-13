#include <iostream>
#include <scilib/mdarray.h>
#include <scilib/linalg.h>

int main()
{
    Sci::Vector<double> x = {1.0, 2.0, 3.0};
    Sci::Vector<double> y = {2.0, 4.0, 6.0};

    std::cout << "x + y = " << x + y << '\n'
              << "dot(x, y) = " << Sci::Linalg::dot(x, y) << '\n';
}
