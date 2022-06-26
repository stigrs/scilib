// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_SUPPORT_H
#define SCILIB_MDARRAY_SUPPORT_H

namespace Sci {
namespace __Detail {

template <class From, class To>
concept convertible_to = std::is_convertible_v<From, To> && requires
{
    static_cast<To>(std::declval<From>());
};

} // namespace __Detail
} // namespace Sci

#endif // SCILIB_MDARRAY_SUPPORT_H
