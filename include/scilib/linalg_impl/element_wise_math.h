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
inline MDArray<T, Extents> abs(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::abs(x); });
    return res;
}

template <class T, class Extents>
    requires std::is_floating_point_v<T> && Scilib::Extents_has_rank<Extents> 
inline MDArray<T, Extents> pow(const MDArray<T, Extents>& m, const T& val)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x, const T& p) { x = std::pow(x, p); }, val);
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> sqrt(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sqrt(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> cbrt(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cbrt(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> exp(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::exp(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> expm1(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::expm1(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> log(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> log10(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log10(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> log2(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::log2(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> erf(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::erf(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> erfc(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::erfc(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> tgamma(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tgamma(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> lgamma(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::lgamma(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> sin(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sin(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> cos(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cos(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> tan(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tan(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> asin(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::asin(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> acos(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::acos(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> atan(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::atan(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> sinh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::sinh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> cosh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::cosh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> tanh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::tanh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> asinh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::asinh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> acosh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::acosh(x); });
    return res;
}

template <class T, class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<T, Extents> atanh(const MDArray<T, Extents>& m)
{
    MDArray<T, Extents> res(m);
    res.apply([](T& x) { x = std::atanh(x); });
    return res;
}

// Complex conjugate.
template <class Extents>
    requires Scilib::Extents_has_rank<Extents>
inline MDArray<std::complex<double>, Extents>
conj(const MDArray<std::complex<double>, Extents>& m)
{
    MDArray<std::complex<double>, Extents> res(m);
    res.apply([](std::complex<double>& x) { x = std::conj(x); });
    return res;
}
// clang-format on

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_ELEMENT_WISE_MATH_H
