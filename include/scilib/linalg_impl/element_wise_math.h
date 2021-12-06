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

namespace Scilib {
namespace Linalg {

//------------------------------------------------------------------------------
//
// Miscellaneous element-wise functions:

// clang-format off
template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> abs(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::abs(x); });
    return res;
}

template <class T, class Extents>
    requires std::is_floating_point_v<T> && Scilib::Extents_has_rank<Extents> 
inline Scilib::MDArray<T, Extents> 
pow(const Scilib::MDArray<T, Extents>& m, const T& val)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x, const T& p) { x = std::pow(x, p); }, val);
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> sqrt(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sqrt(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> cbrt(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cbrt(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> exp(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::exp(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> expm1(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::expm1(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> log(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> log10(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log10(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> log2(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log2(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> erf(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::erf(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> erfc(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::erfc(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> tgamma(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tgamma(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> lgamma(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::lgamma(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> sin(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sin(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> cos(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cos(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> tan(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tan(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> asin(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::asin(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> acos(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::acos(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> atan(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::atan(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> sinh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sinh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> cosh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cosh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> tanh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tanh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> asinh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::asinh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> acosh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::acosh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<T, Extents> atanh(const Scilib::MDArray<T, Extents>& m)
{
    Scilib::MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::atanh(x); });
    return res;
}

// Complex conjugate.
template <class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline Scilib::MDArray<std::complex<double>, Extents>
conj(const Scilib::MDArray<std::complex<double>, Extents>& m)
{
    Scilib::MDArray<std::complex<double>, Extents> res(m);
    res.apply([](std::complex<double>& x) { x = std::conj(x); });
    return res;
}
// clang-format on

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_ELEMENT_WISE_MATH_H
