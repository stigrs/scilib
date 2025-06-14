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

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>

namespace Mdspan = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace Sci {

template <class T,
          class Extents = Mdspan::extents<index, Mdspan::dynamic_extent>,
          class Layout = Mdspan::layout_right,
          class Accessor = Mdspan::default_accessor<T>>
    requires __Detail::Is_extents_v<Extents>
class MDSpan_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<T>;
    using extents_type = Extents;
    using mdspan_type = Mdspan::mdspan<T, Extents, Layout, Accessor>;
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    using difference_type = std::ptrdiff_t;
    using reference = typename mdspan_type::reference;
    using data_handle_type = typename mdspan_type::data_handle_type;

    // Needed for LegacyForwardIterator
    MDSpan_iterator() = default;

    constexpr explicit MDSpan_iterator(mdspan_type x) : x_(x), current_(0), size_(x.extent(0))
    {
        static_assert(Extents::rank() == 1);
    }

    constexpr explicit MDSpan_iterator(mdspan_type x, difference_type curr_index)
        : x_(x), current_(curr_index), size_(x.extent(0))
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
            assert(size_ - current_ >= n);
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
            assert(size_ - current_ >= -n);
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

    friend constexpr iterator operator+(difference_type n, iterator other) noexcept
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
        assert(x_.data_handle() == it.x_.data_handle() && x_.extent(0) == it.x_.extent(0));
        return current_ - it.current_;
    }

    constexpr bool operator==(iterator other) const noexcept
    {
        return current_ == other.current_ && x_.data_handle() == other.x_.data_handle();
    }

    constexpr bool operator!=(iterator other) const noexcept
    {
        return current_ != other.current_ || x_.data_handle() != other.x_.data_handle();
    }

    constexpr bool operator<(iterator other) const noexcept { return current_ < other.current_; }

    constexpr bool operator>(iterator other) const noexcept { return other < *this; }

    constexpr bool operator<=(iterator other) const noexcept { return !(other < *this); }

    constexpr bool operator>=(iterator other) const noexcept { return !(*this < other); }

    constexpr reference operator[](difference_type i) const noexcept
    {
        assert(0 <= i && i < size_);
        return x_(i);
    }

    constexpr reference operator*() const noexcept
    {
        assert(0 <= current_ && current_ < size_);
        return x_[current_];
    }

    constexpr data_handle_type operator->() const noexcept
    {
        assert(0 <= current_ && current_ < size_);
        return x_.accessor().offset(x_.data_handle(), x_.mapping()(current_));
    }

private:
    mdspan_type x_;
    difference_type current_ = 0;
    difference_type size_ = 0;
};

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<T, Extents, Layout, Accessor> begin(Mdspan::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    return iterator(x);
}

template <class T, class Extents, class Layout, class Accessor>
MDSpan_iterator<T, Extents, Layout, Accessor> end(Mdspan::mdspan<T, Extents, Layout, Accessor> x)
{
    using iterator = MDSpan_iterator<T, Extents, Layout, Accessor>;
    return iterator(x, x.extent(0));
}

} // namespace Sci

#endif // SCILIB_MDARRAY_MDSPAN_ITERATOR_H
