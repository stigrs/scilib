#include <scilib/linalg.h>
#include <scilib/mdarray.h>

int main()
{
    using namespace Sci;
    using namespace Sci::Linalg;

    Matrix<double, Mdspan::layout_left> M = {{0.0, -1.0}, {1.0, 0.0}};
    std::cout << "M = \n" << M << "\n\n";

    auto res = matrix_power(M, 0);
    std::cout << "matrix_power(M, 0) = \n" << res << "\n\n";

    res = matrix_power(M, 1);
    std::cout << "matrix_power(M, 1) = \n" << res << "\n\n";

    res = matrix_power(M, 2);
    std::cout << "matrix_power(M, 2) = \n" << res << "\n\n";

    res = matrix_power(M, 3);
    std::cout << "matrix_power(M, 3) = \n" << res << "\n\n";

    res = matrix_power(M, -3);
    std::cout << "matrix_power(M, -3) = \n" << res << "\n\n";

    res = matrix_power(M, 6);
    std::cout << "matrix_power(M, 6) = \n" << res << "\n\n";
}
