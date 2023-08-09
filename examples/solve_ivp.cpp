#include <format>
#include <iostream>
#include <scilib/integrate.h>
#include <scilib/mdarray.h>

Sci::Vector<double> lorentz(double /* t */, const Sci::Vector<double>& y)
{
    Sci::Vector<double> ydot(3);

    const double sigma = 10.0;
    const double R = 28.0;
    const double b = 8.0 / 3.0;

    ydot(0) = sigma * (y(1) - y(0));
    ydot(1) = R * y(0) - y(1) - y(0) * y(2);
    ydot(2) = -b * y(2) + y(0) * y(1);

    return ydot;
}

int main()
{
    Sci::Vector<double> y(3);
    y(0) = 10.0;
    y(1) = 1.0;
    y(2) = 1.0;

    double t0 = 0.0;
    double tf = 0.1;

    std::cout << std::format("At t = {0:.1f}, y = {1:10.6f} {2:10.6f} {3:10.6f}\n", t0, y[0], y[1],
                             y[2]);
    for (int i = 0; i < 10; ++i) {
        Sci::Integrate::solve_ivp(lorentz, t0, tf, y);
        tf += 0.1;
        std::cout << std::format("At t = {0:.1f}, y = {1:10.6f} {2:10.6f} {3:10.6f}\n", t0, y[0],
                                 y[1], y[2]);
    }
}