// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_MKL_ALLOCATOR_H
#define SCILIB_MDARRAY_MKL_ALLOCATOR_H

#include <cstddef>
#include <mkl.h>

#ifndef MKL_MEM_ALIGNMENT
#define MKL_MEM_ALIGNMENT 64
#endif

namespace Sci {

template <class T>
class MKL_allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    constexpr MKL_allocator() noexcept {}
    constexpr MKL_allocator(const MKL_allocator&) noexcept {}

    template <class U>
    constexpr MKL_allocator(const MKL_allocator<U>&) noexcept
    {
    }

    ~MKL_allocator() {}

    constexpr T* allocate(size_type n)
    {
        return (T*) mkl_calloc(n, sizeof(value_type), MKL_MEM_ALIGNMENT);
    }

    constexpr void deallocate(T* p, size_type /* n */) { mkl_free(p); }
};

template <class T1, class T2>
constexpr bool operator==(const MKL_allocator<T1>&,
                          const MKL_allocator<T2>&) noexcept
{
    return true;
}

template <class T1, class T2>
constexpr bool operator!=(const MKL_allocator<T1>& lhs,
                          const MKL_allocator<T2>& rhs) noexcept
{
    return !(lhs == rhs);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_MKL_ALLOCATOR_H
