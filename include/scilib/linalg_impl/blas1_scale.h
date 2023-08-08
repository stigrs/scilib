// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_SCALE_H
#define SCILIB_LINALG_BLAS1_SCALE_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T, class IndexType, std::size_t ext, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline void scale(const T& scalar,
                  Sci::MDArray<T, stdex::extents<IndexType, ext>, Layout, Container>& m)
{
    std::experimental::linalg::scale(scalar, m.to_mdspan());
}

template <class T,
          class IndexType,
          std::size_t numrows,
          std::size_t numcols,
          class Layout,
          class Container>
    requires(std::is_integral_v<IndexType>)
inline void
scale(const T& scalar,
      Sci::MDArray<T, stdex::extents<IndexType, numrows, numcols>, Layout, Container>& m)
{
    std::experimental::linalg::scale(scalar, m.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_SCALE_H
