// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_ELEMENT_WISE_MATH_H
#define SCILIB_LINALG_ELEMENT_WISE_MATH_H

#include <scilib/mdarray.h>
#include <cmath>
#include <complex>
#include <type_traits>

namespace Sci {
namespace Linalg {

//------------------------------------------------------------------------------
//
// Miscellaneous element-wise functions:

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
abs(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::abs(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(std::is_floating_point_v<T>&& Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
pow(const Sci::MDArray<T, Extents, Layout, Allocator>& m, const T& val)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x, const T& p) { x = std::pow(x, p); }, val);
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
sqrt(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::sqrt(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
cbrt(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::cbrt(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
exp(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::exp(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
expm1(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::expm1(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
log(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::log(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
log10(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::log10(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
log2(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::log2(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
erf(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::erf(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
erfc(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::erfc(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
tgamma(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::tgamma(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
lgamma(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::lgamma(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
sin(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::sin(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
cos(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::cos(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
tan(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::tan(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
asin(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::asin(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
acos(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::acos(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
atan(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::atan(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
sinh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::sinh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
cosh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::cosh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
tanh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::tanh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
asinh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::asinh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
acosh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::acosh(x); });
    return res;
}

template <class T, class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<T, Extents, Layout, Allocator>
atanh(const Sci::MDArray<T, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<T, Extents, Layout, Allocator> res(m);
    res.apply([](T& x) { x = std::atanh(x); });
    return res;
}

// Complex conjugate.
template <class Extents, class Layout, class Allocator>
    requires(Sci::Extents_has_rank<Extents>)
inline Sci::MDArray<std::complex<double>, Extents, Layout, Allocator>
conj(const Sci::MDArray<std::complex<double>, Extents, Layout, Allocator>& m)
{
    Sci::MDArray<std::complex<double>, Extents, Layout, Allocator> res(m);
    res.apply([](std::complex<double>& x) { x = std::conj(x); });
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_ELEMENT_WISE_MATH_H
