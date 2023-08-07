// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SWAP_ELEMENTS_H
#define SCILIB_MDARRAY_SWAP_ELEMENTS_H

#include <gsl/gsl>
#include <type_traits>
#include <utility>


namespace Sci {

namespace stdex = std::experimental;

template <class T_x,
          class Extent_x,
          class Layout_x,
          class Accessor_x,
          class Extent_y,
          class T_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y>)
inline void swap_elements(stdex::mdspan<T_x, Extent_x, Layout_x, Accessor_x> x,
                          stdex::mdspan<T_y, Extent_y, Layout_y, Accessor_y> y)
{
    using IndexType_x = typename Extent_x::index_type;
    using IndexType_y = typename Extent_y::index_type;
    using index_type = std::common_type_t<IndexType_x, IndexType_y>;

    for (std::size_t r = 0; r < x.rank(); ++r) {
        Expects(gsl::narrow_cast<index_type>(x.extent(r)) ==
                gsl::narrow_cast<index_type>(y.extent(r)));
    }
    auto swap_fn = [&]<class... IndexTypes>(IndexTypes... indices) {
#if MDSPAN_USE_BRACKET_OPERATOR
        std::swap(x[gsl::narrow_cast<index_type>(std::move(indices))...],
                  y[gsl::narrow_cast<index_type>(std::move(indices))...]);
#else
        std::swap(x(gsl::narrow_cast<index_type>(std::move(indices))...),
                  y(gsl::narrow_cast<index_type>(std::move(indices))...));
#endif
    }; 
    for_each_in_extents(swap_fn, x);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_SWAP_ELEMENTS_H
