// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_SCALED_H
#define SCILIB_LINALG_SCALED_H

#include <experimental/mdspan>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class Reference, class ScalingFactor>
class scaled_scalar {
private:
    Reference value;
    const ScalingFactor scaling_factor;
    using result_type = decltype(value * scaling_factor);

public:
    scaled_scalar(Reference v, const ScalingFactor& s)
        : value(v), scaling_factor(s)
    {
    }

    operator result_type() const { return value * scaling_factor; }
};

template <class Accessor, class S>
class accessor_scaled {
public:
    using element_type = typename Accessor::element_type;
    using pointer = typename Accessor::pointer;
    using reference = scaled_scalar<typename Accessor::reference, S>;
    using offset_policy = accessor_scaled<typename Accessor::offset_policy, S>;

    accessor_scaled() = default;

    accessor_scaled(Accessor a, S sval) : acc_(a), scale_factor_(sval) {}

    reference access(pointer p, stdex::extents<>::size_type i) const noexcept
    {
        return reference(acc_.access(p, i), scale_factor_);
    }

    typename offset_policy::pointer
    offset(pointer p, stdex::extents<>::size_type i) const noexcept
    {
        return acc_.offset(p, i);
    }

    element_type* decay(pointer p) const noexcept { return acc_.decay(p); }

    Accessor nested_accessor() const { return acc_; }

    S scale_factor() const { return scale_factor_; }

private:
    Accessor acc_;
    S scale_factor_;
};

template <class T,
          class Extents,
          class Layout,
          class Accessor,
          class ScalingFactorType>
inline stdex::
    mdspan<T, Extents, Layout, accessor_scaled<Accessor, ScalingFactorType>>
    scaled(ScalingFactorType scaling_factor,
           stdex::mdspan<T, Extents, Layout, Accessor> a)
{
    using accessor_t = accessor_scaled<Accessor, ScalingFactorType>;
    return stdex::mdspan<T, Extents, Layout, accessor_t>(
        a.data(), a.mapping(), accessor_t(a.accessor(), scaling_factor));
}

template <class T, class Extents, class Layout, class ScalingFactorType>
inline Sci::MDArray<T, Extents, Layout>
scaled(ScalingFactorType scaling_factor,
       const Sci::MDArray<T, Extents, Layout>& a)
{
    Sci::MDArray<T, Extents, Layout> tmp = scaled(scaling_factor, a.view());
    return tmp;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_SCALED_H
