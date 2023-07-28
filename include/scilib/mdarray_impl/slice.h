// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SLICE_H
#define SCILIB_MDARRAY_SLICE_H

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace Sci {
namespace stdex = std::experimental;

// Generate a tuple for slicing.
inline std::tuple<std::size_t, std::size_t> seq(std::size_t first, std::size_t last)
{
    return std::tuple<std::size_t, std::size_t>{first, last};
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto first(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v,
                  std::size_t count)
{
    std::pair<std::size_t, std::size_t> slice{0, count};
    return stdex::submdspan(v, slice);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto first(MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v,
                  std::size_t count)
{
    return first(v.to_mdspan(), count);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto first(const MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v,
                  std::size_t count)
{
    return first(v.to_mdspan(), count);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto last(stdex::mdspan<T, stdex::extents<IndexType, ext>, Layout, Accessor> v,
                 std::size_t count)
{
    std::pair<std::size_t, std::size_t> slice{v.extent(0) - count, v.extent(0)};
    return stdex::submdspan(v, slice);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto last(MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v,
                 std::size_t count)
{
    return last(v.to_mdspan(), count);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto last(const MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& v,
                 std::size_t count)
{
    return last(v.to_mdspan(), count);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto row(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> m,
                std::size_t i)
{
    return stdex::submdspan(m, i, stdex::full_extent);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline auto row(MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& m,
                std::size_t i)
{
    return row(m.to_mdspan(), i);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline auto row(const MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& m,
                std::size_t i)
{
    return row(m.to_mdspan(), i);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto column(stdex::mdspan<T, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> m,
                   std::size_t j)
{
    return stdex::submdspan(m, stdex::full_extent, j);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline auto column(MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& m,
                   std::size_t i)
{
    return column(m.to_mdspan(), i);
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline auto column(const MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& m,
                   std::size_t i)
{
    return column(m.to_mdspan(), i);
}

template <class T, class IndexType, std::size_t ext, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline auto diag(stdex::mdspan<T, stdex::extents<IndexType, ext, ext>, Layout, Accessor> m)
{
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        return stdex::mdspan<T, stdex::extents<index, stdex::dynamic_extent>, stdex::layout_stride>{
            m.data_handle(),
            {stdex::dextents<index, 1>{m.extent(0)}, std::array<index, 1>{m.stride(1) + 1}}};
    }
    else {
        return stdex::mdspan<T, stdex::extents<index, stdex::dynamic_extent>, stdex::layout_stride>{
            m.data_handle(),
            {stdex::dextents<index, 1>{m.extent(0)}, std::array<index, 1>{m.stride(0) + 1}}};
    }
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto diag(MDArray<T, stdex::extents<IndexType, ext, ext>, Layout, Container>& m)
{
    return diag(m.to_mdspan());
}

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline auto diag(const MDArray<T, stdex::extents<IndexType, ext, ext>, Layout, Container>& m)
{
    return diag(m.to_mdspan());
}

template <class T, class Extents, class Layout, class Accessor, class... SliceSpecs>
// Check of SliceSpecs is done by submdspan
inline auto slice(stdex::mdspan<T, Extents, Layout, Accessor> m, SliceSpecs... slices)
{
    return stdex::submdspan(m, slices...);
}

template <class T, class Extents, class Layout, class Container, class... SliceSpecs>
inline auto slice(MDArray<T, Extents, Layout, Container>& m, SliceSpecs... slices)
{
    return slice(m.to_mdspan(), slices...);
}

template <class T, class Extents, class Layout, class Container, class... SliceSpecs>
inline auto slice(const MDArray<T, Extents, Layout, Container>& m, SliceSpecs... slices)
{
    return slice(m.to_mdspan(), slices...);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SLICE_H