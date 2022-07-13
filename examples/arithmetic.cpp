#include <iostream>
#include <scilib/mdarray.h>

int main()
{
    Sci::Vector<double> x = {1.0, 2.0, 3.0};
    Sci::Vector<double> y = {2.0, 4.0, 6.0};

    std::cout << 2.0 * x + y << '\n';
}
