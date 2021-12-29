// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

// Based on code from:
//
//     https://github.com/kokkos/stdBLAS/blob/main/tests/iterator.cpp
//
// Copyright (2019) Sandia Corporation

#ifndef SCILIB_MDARRAY_MDSPAN_ITERATOR_H
#define SCILIB_MDARRAY_MDSPAN_ITERATOR_H

#include <experimental/mdspan>
#include <cstddef>
#include <cassert>
#include <iterator>
#include <iostream>

namespace Scilib {

namespace stdex = std::experimental;

// clang-format off
template <class T,
          class Extents = stdex::extents<stdex::dynamic_extent>,
          class Layout = stdex::layout_right,
          class Accessor = stdex::default_accessor<T>>
    requires Extents_has_rank<Extents>
class MDSpan_iterator {
public:
    // clang-format on
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<T>;
    using extents_t = Extents;
    using mdspan_t = stdex::mdspan<T, Extents, Layout, Accessor>;
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    using difference_type = typename mdspan_t::difference_type;
    using reference = typename mdspan_t::reference;
    using pointer = typename mdspan_t::pointer;

    // Needed for LegacyForwardIterator
    MDSpan_iterator() = default;

    constexpr explicit MDSpan_iterator(mdspan_t x)
        : x_(x), current_(0), end_(x.extent(0))
    {
        static_assert(Extents::rank() == 1);
    }

    constexpr explicit MDSpan_iterator(mdspan_t x, difference_type curr_index)
        : x_(x), current_(curr_index), end_(x.extent(0))
    {
        static_assert(Extents::rank() == 1);
    }

    constexpr iterator& operator++() noexcept
    {
        ++current_;
        return *this;
    }

    constexpr iterator operator++(int) noexcept
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    constexpr iterator& operator--() noexcept
    {
        --current_;
        return *this;
    }

    constexpr iterator operator--(int) noexcept
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    constexpr iterator& operator+=(difference_type n) noexcept
    {
        if (n > 0) {
            assert(end_ - current_ >= n);
        }
        if (n < 0) {
            assert(current_ >= -n);
        }
        current_ += n;
        return *this;
    }

    constexpr iterator& operator-=(difference_type n) noexcept
    {
        if (n > 0) {
            assert(current_ >= n);
        }
        if (n < 0) {
            assert(end_ - current_ >= -n);
        }
        current_ -= n;
        return *this;
    }

    constexpr iterator operator+(difference_type n) const noexcept
    {
        auto tmp = *this;
        tmp += n;
        return tmp;
    }

    friend constexpr iterator operator+(difference_type n,
                                        iterator other) noexcept
    {
        return other + n;
    }

    constexpr iterator operator-(difference_type n) const noexcept
    {
        auto tmp = *this;
        tmp -= n;
        return tmp;
    }

    constexpr difference_type operator-(iterator it) const noexcept
    {
        assert(x_.data() == it.x_.data() && x_.extent(0) == it.x_.extent(0));
        return current_ - it.current_;
    }

    constexpr bool operator==(iterator other) const noexcept
    {
        return current_ == other.current_ && x_.data() == other.x_.data();
    }

    constexpr bool operator!=(iterator other) const noexcept
    {
        return current_ != other.current_ || x_.data() != other.x_.data();
    }

    constexpr bool operator<(iterator other) const noexcept
    {
        return current_ < other.current_;
    }

    constexpr bool operator>(iterator other) const noexcept
    {
        return other < *this;
    }

    constexpr bool operator<=(iterator other) const noexcept
    {
        return !(other < *this);
    }

    constexpr bool operator>=(iterator other) const noexcept
    {
        return !(*this < other);
    }

    constexpr reference operator[](difference_type i) const noexcept
    {
        assert(0 <= i && i < end_);
        return x_(i);
    }

    constexpr reference operator*() const noexcept
    {
        assert(0 <= current_ && current_ < end_);
        return x_(current_);
    }

    constexpr pointer operator->() const noexcept
    {
        assert(0 <= current_ && current_ < end_);
        return x_.accessor().offset(x_.data(), x_.mapping()(current_));
    }

private:
    mdspan_t x_;
    difference_type current_ = 0;
    difference_type end_ = 0;
};

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<T, Extents, Layout, Accessor>
begin(stdex::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    return iterator(x);
}

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<const T, Extents, Layout, Accessor>
cbegin(stdex::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<const T, Extents, Layout, Accessor>;
    return iterator(x);
}

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<T, Extents, Layout, Accessor>
end(stdex::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    return iterator(x, x.extent(0));
}

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<const T, Extents, Layout, Accessor>
cend(stdex::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<const T, Extents, Layout, Accessor>;
    return iterator(x, x.extent(0));
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_MDSPAN_ITERATOR_H
