// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SLICE_H
#define SCILIB_MDARRAY_SLICE_H

#include <experimental/mdspan>
#include <array>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace Sci {
namespace stdex = std::experimental;

// Generate a tuple for slicing.
inline std::tuple<std::size_t, std::size_t> seq(std::size_t first,
                                                std::size_t last)
{
    return std::tuple<std::size_t, std::size_t>{first, last};
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto first(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v,
                  stdex::extents<>::size_type count)
{
    using size_type = stdex::extents<>::size_type;
    std::pair<size_type, size_type> slice{0, count};
    return stdex::submdspan(v, slice);
}

template <class T, class Layout>
inline auto first(Vector<T, Layout>& v, stdex::extents<>::size_type count)
{
    return first(v.view(), count);
}

template <class T, class Layout>
inline auto first(const Vector<T, Layout>& v, stdex::extents<>::size_type count)
{
    return first(v.view(), count);
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto last(stdex::mdspan<T, stdex::extents<ext>, Layout, Accessor> v,
                 stdex::extents<>::size_type count)
{
    using size_type = stdex::extents<>::size_type;
    std::pair<size_type, size_type> slice{v.extent(0) - count, v.extent(0)};
    return stdex::submdspan(v, slice);
}

template <class T, class Layout>
inline auto last(Vector<T, Layout>& x, stdex::extents<>::size_type count)
{
    return last(x.view(), count);
}

template <class T, class Layout>
inline auto last(const Vector<T, Layout>& x, stdex::extents<>::size_type count)
{
    return last(x.view(), count);
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline auto
row(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
    stdex::extents<>::size_type i)
{
    return stdex::submdspan(m, i, stdex::full_extent);
}

template <class T, class Layout>
inline auto row(Matrix<T, Layout>& m, stdex::extents<>::size_type i)
{
    return row(m.view(), i);
}

template <class T, class Layout>
inline auto row(const Matrix<T, Layout>& m, stdex::extents<>::size_type i)
{
    return row(m.view(), i);
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout,
          class Accessor>
inline auto
column(stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout, Accessor> m,
       stdex::extents<>::size_type j)
{
    return stdex::submdspan(m, stdex::full_extent, j);
}

template <class T, class Layout>
inline auto column(Matrix<T, Layout>& m, stdex::extents<>::size_type i)
{
    return column(m.view(), i);
}

template <class T, class Layout>
inline auto column(const Matrix<T, Layout>& m, stdex::extents<>::size_type i)
{
    return column(m.view(), i);
}

template <class T,
          stdex::extents<>::size_type ext,
          class Layout,
          class Accessor>
inline auto diag(stdex::mdspan<T, stdex::extents<ext, ext>, Layout, Accessor> m)
{
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        return Subvector_view<T>{m.data(),
                                 {stdex::dextents<1>{m.extent(0)},
                                  std::array<std::size_t, 1>{m.stride(1) + 1}}};
    }
    else {
        return Subvector_view<T>{m.data(),
                                 {stdex::dextents<1>{m.extent(0)},
                                  std::array<std::size_t, 1>{m.stride(0) + 1}}};
    }
}

template <class T, class Layout>
inline auto diag(Matrix<T, Layout>& m)
{
    return diag(m.view());
}

template <class T, class Layout>
inline auto diag(const Matrix<T, Layout>& m)
{
    return diag(m.view());
}

template <class T,
          class Extents,
          class Layout,
          class Accessor,
          class... SliceSpecs>
    // Check of SliceSpecs is done by submdspan
inline auto slice(stdex::mdspan<T, Extents, Layout, Accessor> m,
                  SliceSpecs... slices)
{
    return stdex::submdspan(m, slices...);
}

template <class T, class Extents, class Layout, class... SliceSpecs>
inline auto slice(MDArray<T, Extents, Layout>& m, SliceSpecs... slices)
{
    return slice(m.view(), slices...);
}

template <class T, class Extents, class Layout, class... SliceSpecs>
inline auto slice(const MDArray<T, Extents, Layout>& m, SliceSpecs... slices)
{
    return slice(m.view(), slices...);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SLICE_H