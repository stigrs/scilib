// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SLICE_H
#define SCILIB_MDARRAY_SLICE_H

#include <array>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace Sci {
namespace stdex = std::experimental;

// Generate a tuple for slicing.
inline std::tuple<std::size_t, std::size_t> seq(std::size_t first, std::size_t last)
{
    return std::tuple<std::size_t, std::size_t>{first, last};
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto first(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v,
                  std::size_t count)
{
    using size_type = std::size_t;
    std::pair<size_type, size_type> slice{0, count};
    return stdex::submdspan(v, slice);
}

template <class T, class Layout, class Allocator>
inline auto first(Vector<T, Layout, Allocator>& v, std::size_t count)
{
    return first(v.view(), count);
}

template <class T, class Layout, class Allocator>
inline auto first(const Vector<T, Layout, Allocator>& v, std::size_t count)
{
    return first(v.view(), count);
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto last(stdex::mdspan<T, stdex::extents<std::size_t, ext>, Layout, Accessor> v,
                 std::size_t count)
{
    using size_type = std::size_t;
    std::pair<size_type, size_type> slice{v.extent(0) - count, v.extent(0)};
    return stdex::submdspan(v, slice);
}

template <class T, class Layout, class Allocator>
inline auto last(Vector<T, Layout, Allocator>& x, std::size_t count)
{
    return last(x.view(), count);
}

template <class T, class Layout, class Allocator>
inline auto last(const Vector<T, Layout, Allocator>& x, std::size_t count)
{
    return last(x.view(), count);
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
inline auto row(stdex::mdspan<T, stdex::extents<std::size_t, nrows, ncols>, Layout, Accessor> m,
                std::size_t i)
{
    return stdex::submdspan(m, i, stdex::full_extent);
}

template <class T, class Layout, class Allocator>
inline auto row(Matrix<T, Layout, Allocator>& m, std::size_t i)
{
    return row(m.view(), i);
}

template <class T, class Layout, class Allocator>
inline auto row(const Matrix<T, Layout, Allocator>& m, std::size_t i)
{
    return row(m.view(), i);
}

template <class T, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
inline auto column(stdex::mdspan<T, stdex::extents<std::size_t, nrows, ncols>, Layout, Accessor> m,
                   std::size_t j)
{
    return stdex::submdspan(m, stdex::full_extent, j);
}

template <class T, class Layout, class Allocator>
inline auto column(Matrix<T, Layout, Allocator>& m, std::size_t i)
{
    return column(m.view(), i);
}

template <class T, class Layout, class Allocator>
inline auto column(const Matrix<T, Layout, Allocator>& m, std::size_t i)
{
    return column(m.view(), i);
}

template <class T, std::size_t ext, class Layout, class Accessor>
inline auto diag(stdex::mdspan<T, stdex::extents<std::size_t, ext, ext>, Layout, Accessor> m)
{
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        return Subvector_view<T>{m.data(),
                                 {stdex::dextents<std::size_t, 1>{m.extent(0)},
                                  std::array<std::size_t, 1>{m.stride(1) + 1}}};
    }
    else {
        return Subvector_view<T>{m.data(),
                                 {stdex::dextents<std::size_t, 1>{m.extent(0)},
                                  std::array<std::size_t, 1>{m.stride(0) + 1}}};
    }
}

template <class T, class Layout, class Allocator>
inline auto diag(Matrix<T, Layout, Allocator>& m)
{
    return diag(m.view());
}

template <class T, class Layout, class Allocator>
inline auto diag(const Matrix<T, Layout, Allocator>& m)
{
    return diag(m.view());
}

template <class T, class Extents, class Layout, class Accessor, class... SliceSpecs>
// Check of SliceSpecs is done by submdspan
inline auto slice(stdex::mdspan<T, Extents, Layout, Accessor> m, SliceSpecs... slices)
{
    return stdex::submdspan(m, slices...);
}

template <class T, class Extents, class Layout, class Allocator, class... SliceSpecs>
inline auto slice(MDArray<T, Extents, Layout, Allocator>& m, SliceSpecs... slices)
{
    return slice(m.view(), slices...);
}

template <class T, class Extents, class Layout, class Allocator, class... SliceSpecs>
inline auto slice(const MDArray<T, Extents, Layout, Allocator>& m, SliceSpecs... slices)
{
    return slice(m.view(), slices...);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SLICE_H