//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen Germany
//
/// \file strides.hpp Definition for the basic_strides template class


#ifndef _BOOST_UBLAS_TENSOR_ACCESS_HPP_
#define _BOOST_UBLAS_TENSOR_ACCESS_HPP_

#include "tags.hpp"
#include "../detail/config.hpp"


namespace boost { namespace numeric { namespace ublas {

template<class __int_type, class __layout>
class basic_strides;

namespace detail {


/** @brief Returns relative memory index with respect to a multi-index
 *
 * @code auto j = access(std::vector{3,4,5}, strides{shape{4,2,3},first_order}); @endcode
 *
 * @param[in] i multi-index of length p
 * @param[in] w stride vector of length p
 * @returns relative memory location depending on \c i and \c w
*/
BOOST_UBLAS_INLINE
template<class size_type, class layout_type>
auto access(basic_strides<size_type,layout_type> const& w, std::vector<size_type> const& i)
{
	const auto p = i.size();
	size_type sum = 0u;
	for(auto r = 0u; r < p; ++r)
		sum += i[r]*w[r];
	return sum;
}

/** @brief Returns relative memory index with respect to a multi-index
 *
 * @code auto j = access(0, strides{shape{4,2,3},first_order}, 2,3,4); @endcode
 *
 * @param[in] i   first element of the partial multi-index
 * @param[in] is  the following elements of the partial multi-index
 * @param[in] sum the current relative memory index
 * @returns relative memory location depending on \c i and \c w
*/
BOOST_UBLAS_INLINE
template<std::size_t r, class layout_type, class ... size_types>
auto access(std::size_t sum, basic_strides<std::size_t, layout_type> const& w, std::size_t i, size_types ... is)
{
	sum+=i*w[r];
	if constexpr (sizeof...(is) == 0)
		return sum;
	else
		return detail::access<r+1>(sum,w,std::forward<size_types>(is)...);
}

} // detail
} // ublas
} // numeric
} // boost

#endif
