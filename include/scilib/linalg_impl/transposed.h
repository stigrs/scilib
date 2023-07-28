// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_TRANSPOSED_H
#define SCILIB_LINALG_TRANSPOSED_H

#include <experimental/linalg>
#include <type_traits>

namespace Sci {
namespace Linalg {

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline Sci::MDArray<T, stdex::extents<IndexType, ncols, nrows>, Layout, Container>
transposed(const Sci::MDArray<T, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& a)
{
    return Sci::MDArray<T, stdex::extents<IndexType, ncols, nrows>, Layout, Container>(
        std::experimental::linalg::transposed(a.to_mdspan()));
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_TRANSPOSED_H
