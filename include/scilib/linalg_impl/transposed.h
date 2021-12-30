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

#ifndef SCILIB_LINALG_TRANSPOSED_H
#define SCILIB_LINALG_TRANSPOSED_H

#include <experimental/mdspan>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

namespace {
template <stdex::extents<>::size_type ext0, stdex::extents<>::size_type ext1>
stdex::extents<ext1, ext0> transpose_extents(stdex::extents<ext0, ext1> e)
{
    return stdex::extents<ext1, ext0>(e.extent(1), e.extent(0));
}
} // namespace

template <class Layout>
class layout_transpose {
public:
    template <class Extents>
    struct mapping {
    private:
        using nested_mapping_type = typename Layout::template mapping<Extents>;

    public:
        using size_type = typename Extents::size_type;

        nested_mapping_type nested_mapping;

        mapping() = default;

        mapping(nested_mapping_type map) : nested_mapping(map) {}

        // for non-batched layouts
        size_type operator()(size_type i, size_type j) const
        {
            return nested_mapping(j, i);
        }

        constexpr auto extents() const noexcept
        {
            return transpose_extents(nested_mapping.extents());
        }

        constexpr bool is_unique() const noexcept
        {
            return nested_mapping.is_unique();
        }
        constexpr bool is_contiguous() const noexcept
        {
            return nested_mapping.is_contiguous();
        }
        constexpr bool is_strided() const noexcept
        {
            return nested_mapping.is_strided();
        }

        constexpr size_type stride(size_type r) const noexcept
        {
            return nested_mapping.stride(size_type(1) - r);
        }
    };
};

template <class T, class Extents, class Layout, class Accessor>
stdex::mdspan<T, Extents, layout_transpose<Layout>, Accessor>
transposed(stdex::mdspan<T, Extents, Layout, Accessor> a)
{
    static_assert(a.rank() == 2);
    using layout_type = layout_transpose<Layout>;
    using mapping_type = typename layout_type::template mapping<Extents>;
    return stdex::mdspan<T, Extents, layout_type, Accessor>(
        a.data(), mapping_type(a.mapping()), a.accessor());
}

template <class T, class Layout>
inline Scilib::Matrix<T, Layout> transposed(const Scilib::Matrix<T, Layout>& a)
{
    Scilib::Matrix<T, Layout> tmp = transposed(a.view());
    return tmp;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_TRANSPOSED_H
