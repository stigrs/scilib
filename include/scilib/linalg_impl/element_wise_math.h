// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_ELEMENT_WISE_MATH_H
#define SCILIB_LINALG_ELEMENT_WISE_MATH_H

#include <cmath>
#include <complex>
#include <type_traits>

namespace Sci {
namespace Linalg {

//--------------------------------------------------------------------------------------------------
//
// Miscellaneous element-wise functions:

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
abs(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::abs(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
    requires(std::is_floating_point_v<T>)
inline Sci::MDArray<T, Extents, Layout, Container>
pow(const Sci::MDArray<T, Extents, Layout, Container>& m, const T& val)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x, const T& p) { x = std::pow(x, p); }, val);
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
sqrt(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::sqrt(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
cbrt(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::cbrt(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
exp(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::exp(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
expm1(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::expm1(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
log(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::log(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
log10(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::log10(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
log2(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::log2(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
erf(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::erf(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
erfc(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::erfc(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
tgamma(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::tgamma(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
lgamma(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::lgamma(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
sin(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::sin(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
cos(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::cos(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
tan(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::tan(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
asin(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::asin(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
acos(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::acos(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
atan(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::atan(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
sinh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::sinh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
cosh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::cosh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
tanh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::tanh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
asinh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::asinh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
acosh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::acosh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Container>
inline Sci::MDArray<T, Extents, Layout, Container>
atanh(const Sci::MDArray<T, Extents, Layout, Container>& m)
{
    Sci::MDArray<T, Extents, Layout, Container> res(m);
    res.apply([](T& x) { x = std::atanh(x); });
    return res;
}

// Complex conjugate.
template <class Extents, class Layout, class Container>
inline Sci::MDArray<std::complex<double>, Extents, Layout, Container>
conj(const Sci::MDArray<std::complex<double>, Extents, Layout, Container>& m)
{
    Sci::MDArray<std::complex<double>, Extents, Layout, Container> res(m);
    res.apply([](std::complex<double>& x) { x = std::conj(x); });
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_ELEMENT_WISE_MATH_H
