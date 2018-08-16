//  Copyright (c) 2018
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which firsted as a Google Summer of Code project.
//


/// \file subtensor_utility.hpp Definition for the tensor template class

#ifndef _BOOST_NUMERIC_UBLAS_SUBTENSOR_UTILITY_HPP_
#define _BOOST_NUMERIC_UBLAS_SUBTENSOR_UTILITY_HPP_

#include "span.hpp"
#include "extents.hpp"
#include "tags.hpp"


namespace boost { namespace numeric { namespace ublas { namespace detail {


/*! @brief Computes span strides for a subtensor
 *
 * span stride v is computed according to: v[i] = w[i]*s[i], where
 * w[i] is the i-th stride of the tensor
 * s[i] is the step size of the i-th span
 *
 * @param[in] strides strides of the tensor, the subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<class strides_type, class span_type>
auto span_strides(strides_type const& strides, std::vector<span_type> const& spans)
{
	if(strides.size() != spans.size())
		throw std::runtime_error("Error in boost::numeric::ublas::subtensor::span_strides(): tensor strides.size() != spans.size()");

	using base_type = typename strides_type::base_type;
	auto span_strides = base_type(spans.size());

	std::transform(strides.begin(), strides.end(), spans.begin(), span_strides.begin(),
								 [](auto w, auto const& s) { return w * s.step(); } );

	return strides_type( span_strides );
}

/*! @brief Computes the data pointer offset for a subtensor
 *
 * offset is computed according to: sum ( f[i]*w[i] ), where
 * f[i] is the first element of the i-th span
 * w[i] is the i-th stride of the tensor
 *
 * @param[in] strides strides of the tensor, the subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<class strides_type, class span_type>
auto offset(strides_type const& strides, std::vector<span_type> const& spans)
{
	if(strides.size() != spans.size())
		throw std::runtime_error("Error in boost::numeric::ublas::subtensor::offset(): tensor strides.size() != spans.size()");

	using value_type = typename strides_type::value_type;

	return std::inner_product(spans.begin(), spans.end(), strides.begin(), value_type(0),
														std::plus<value_type>(), [](span_type const& s, value_type w) {return s.first() * w; } );
}


/*! @brief Computes the extents of the subtensor.
 *
 * i-th extent is given by span[i].size()
 *
 * @param[in] spans vector of spans of the subtensor
 */
template<class span_type>
auto extents(std::vector<span_type> const& spans)
{
	using base_type  = typename shape::base_type;
	if(spans.empty())
		return shape{};
	auto extents = base_type(spans.size());
	std::transform(spans.begin(), spans.end(), extents.begin(), [](span_type const& s) { return s.size(); } );
	return shape( extents );
}


/*! @brief Auxiliary function for subtensor which possibly transforms a span instance
 *
 * transform_span(span()     ,4) -> span(0,3)
 * transform_span(span(1,1)  ,4) -> span(1,1)
 * transform_span(span(1,3)  ,4) -> span(1,3)
 * transform_span(span(2,end),4) -> span(2,3)
 * transform_span(span(end)  ,4) -> span(3,3)
 *
 * @note span is zero-based indexed.
 *
 * @param[in] s      span that is going to be transformed
 * @param[in] extent extent that is maybe used for the tranformation
 */
template<class size_type, class span_tag>
auto transform_span(span<span_tag, size_type> const& s, size_type const extent)
{
	using span_type = span<span_tag, size_type>;

	size_type first = s.first();
	size_type last  = s.last ();
	size_type size  = s.size ();

	auto const extent0 = extent-1;

	auto constexpr is_sliced = std::is_same<span_tag,boost::numeric::ublas::tag::sliced>::value;


	if constexpr ( is_sliced ){
		if(size == 0)        return span_type(0       , extent0);
		else if(first== end) return span_type(extent0 , extent0);
		else if(last == end) return span_type(first   , extent0);
		else                 return span_type(first   , last  );
	}
	else {
		size_type step  = s.step ();
		if(size == 0)        return span_type(0       , size_type(1), extent0);
		else if(first== end) return span_type(extent0 , step, extent0);
		else if(last == end) return span_type(first   , step, extent0);
		else                 return span_type(first   , step, last  );
	}
}

template<std::size_t r, class extents_type, class size_type, class span_type, class ... span_types>
void transform_spans_impl (extents_type const& extents,
													 std::vector<span_type>& span_vector,
													 size_type arg,
													 span_types&& ... spans );

template<std::size_t r,class extents_type, class span_type, class ... span_types>
void transform_spans_impl(extents_type const& extents,
													std::vector<span_type>& span_vector,
													span_type const& s,
													span_types&& ... spans)
{
	span_vector.at(r) = transform_span(s,extents.at(r));
	if constexpr (sizeof...(spans)>0)
		transform_spans_impl<r+1>(extents, span_vector, std::forward<span_types>(spans)...);
}

template<std::size_t r, class extents_type, class size_type, class span_type, class ... span_types>
void transform_spans_impl (extents_type const& extents,
													 std::vector<span_type>& span_vector,
													 size_type arg,
													 span_types&& ... spans )
{
	span_vector.at(r) = transform_span(span_type(arg),extents.at(r));
	if constexpr (sizeof...(spans)>0)
		transform_spans_impl<r+1>(extents, span_vector, std::forward<span_types>(spans) ... );
}


/*! @brief Auxiliary function for subtensor that generates vector of spans
 *
 * generate_span_vector<span>(shape(4,3,5,2), span(), 1, span(2,end), end  )
 * -> vector (span(0,3), span(1,1), span(2,4),span(1,1))
 *
 * @note span is zero-based indexed.
 *
 * @param[in] extents of the tensor
 * @param[in] spans spans with which the subtensor is created
 */
template<class span_type, class ... span_types>
auto generate_span_vector(shape const& s, span_types&& ... spans)
{
	constexpr auto n = sizeof...(spans);
	if(s.size() != n)
		throw std::runtime_error("Error in boost::numeric::ublas::generate_span_vector() when creating subtensor: the number of spans does not match with the tensor rank.");
	std::vector<span_type> span_vector(n);
	if constexpr (n>0)
		transform_spans_impl<n-n>(  s, span_vector, std::forward<span_types>(spans)... );
	return span_vector;
}



} // namespace detail
} // namespace ublas
} // namespace numeric
} // namespace boost





#endif
