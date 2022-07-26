#define EIGEN_STACK_ALLOCATION_LIMIT 10000000000000

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5054 4305)
#include <Eigen/Core>
#pragma warning(pop)
#else
#include <Eigen/Core>
#endif

#include <chrono>
#include <cmath>
#include <iostream>
#include <scilib/linalg.h>
#include <scilib/mdarray.h>

typedef std::chrono::duration<double, std::milli> Timer;

template <typename T, int num>
T finite_difference_impl(Sci::StaticMatrix<T, num, num>& u)
{
    using namespace Sci;

    auto u_old = u;

    auto u_old1 = slice(u_old, seq(0, num - 3), seq(1, num - 2));
    auto u_old2 = slice(u_old, seq(2, num - 1), seq(1, num - 2));
    auto u_old3 = slice(u_old, seq(1, num - 2), seq(0, num - 3));
    auto u_old4 = slice(u_old, seq(1, num - 2), seq(2, num - 1));
    auto u_old5 = slice(u_old, seq(0, num - 3), seq(0, num - 3));
    auto u_old6 = slice(u_old, seq(0, num - 3), seq(2, num - 1));
    auto u_old7 = slice(u_old, seq(2, num - 1), seq(0, num - 3));
    auto u_old8 = slice(u_old, seq(2, num - 1), seq(2, num - 1));

    auto u_span = slice(u, seq(1, num - 2), seq(1, num - 2));

    for (int i = 0; i < u_span.extent(0); ++i) {
        for (int j = 0; j < u_span.extent(1); ++j) {
            u_span(i, j) = ((u_old1(i, j) + u_old2(i, j) + u_old3(i, j) + u_old4(i, j)) * 4.0 +
                            u_old5(i, j) + u_old6(i, j) + u_old7(i, j) + u_old8(i, j)) /
                           20.0;
        }
    }
    return Linalg::matrix_norm(u - u_old, 'F');
}

template <typename T, int num>
void run_finite_difference()
{

    T pi = 4.0 * std::atan(1.0);
    T err = 2.0;
    int iter = 0;

    Sci::StaticVector<T, num> x;
    for (int i = 0; i < num; ++i) {
        x(i) = i * pi / (num - 1);
    }

    Sci::StaticMatrix<T, num, num> u;
    u = T{0};

    auto u_col1 = column(u, 0);
    for (int i = 0; i < num; ++i) {
        u_col1(i) = std::sin(x(i));
    }
    auto u_col2 = column(u, num - 1);
    for (int i = 0; i < num; ++i) {
        u_col2(i) = std::sin(x(i)) * std::exp(-pi);
    }

    while (iter < 100000 && err > 1e-6) {
        err = finite_difference_impl<T, num>(u);
        iter++;
    }
    std::cout << "Relative error is:    " << err << '\n' << "Number of iterations: " << iter << '\n';
}

template <typename T, int num, int SO, int Rows, int Cols>
T eigen_finite_difference_impl(Eigen::Matrix<T, num, num, SO, Rows, Cols>& u)
{
    using namespace Eigen;
    Eigen::Matrix<T, num, num, SO, Rows, Cols> u_old = u;

    u.block(1, 1, num - 2, num - 2) =
        ((u_old.block(0, 1, num - 2, num - 2) + u_old.block(2, 1, num - 1, num - 2) +
          u_old.block(1, 0, num - 2, num - 3) + u_old.block(1, 2, num - 2, num - 1)) *
             4.0 +
         u_old.block(0, 0, num - 3, num - 3) + u_old.block(0, 2, num - 3, num - 1) +
         u_old.block(2, 0, num - 1, num - 3) + u_old.block(2, 2, num - 1, num - 1)) /
        20.0;

    return (u - u_old).norm();
}

template <typename T, int num>
void eigen_run_finite_difference()
{

    T pi = 4.0 * std::atan(1.0);
    T err = 2.0;
    int iter = 0;

    Eigen::Matrix<T, num, 1> x;
    for (int i = 0; i < num; ++i) {
        x(i) = i * pi / (num - 1);
    }

    Eigen::Matrix<T, num, num> u;
    u.setZero();
    u.col(0) = x.array().sin();
    u.col(num - 1) = x.array().sin() * std::exp(-pi);

    while (iter < 100000 && err > 1e-6) {
        err = eigen_finite_difference_impl(u);
        iter++;
    }
    std::cout << "Relative error is:    " << err << '\n' << "Number of iterations: " << iter << '\n';
}

int main(int argc, char* argv[])
{

    using T = double;
    int N;
    if (argc == 2) {
        N = atoi(argv[1]);
    }
    else {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return -1;
    }

    std::cout << "Scilib run:\n" << "-----------------------------------\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    if (N == 10)
        run_finite_difference<T, 10>();
    if (N == 100)
        run_finite_difference<T, 100>();
    if (N == 150)
        run_finite_difference<T, 150>();
    if (N == 200)
        run_finite_difference<T, 200>();
    auto t2 = std::chrono::high_resolution_clock::now();
    Timer t_scilib = t2 - t1;
    std::cout << "Elapsed time is:      " << t_scilib.count() << " ms\n\n";

    std::cout << "Eigen run:\n" << "-----------------------------------\n";
    t1 = std::chrono::high_resolution_clock::now();
    if (N == 10)
        eigen_run_finite_difference<T, 10>();
    if (N == 100)
        eigen_run_finite_difference<T, 100>();
    if (N == 150)
        eigen_run_finite_difference<T, 150>();
    if (N == 200)
        eigen_run_finite_difference<T, 200>();
    t2 = std::chrono::high_resolution_clock::now();
    Timer t_eigen = t2 - t1;
    std::cout << "Elapsed time is:      " << t_eigen.count() << " ms\n";
}
