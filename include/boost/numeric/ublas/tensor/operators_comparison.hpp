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

#ifndef _BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_
#define _BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_

#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <type_traits>
#include <functional>

namespace boost::numeric::ublas {
template<class element_type, class storage_format, class storage_type>
class tensor;

namespace detail {
/** @brief Evaluates expression for a tensor
 *
 * Applies a unary function to the results of the expressions before the assignment.
 *
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor matches those of all tensors within the expression.
*/
template<class T, class F, class A, class BinaryPred>
bool compare(boost::numeric::ublas::tensor<T,F,A> const& lhs,
			 boost::numeric::ublas::tensor<T,F,A> const& rhs,
			 BinaryPred pred)
{
	for(auto i = 0u; i < lhs.size(); ++i)
		if(!pred(lhs(i), rhs(i)))
			return false;
	return true;
}


template<class T, class L, class R, class BinaryPred>
bool compare(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs,
			 BinaryPred pred)
{
	constexpr auto lhs_is_tensor = std::is_same<T,L>::value;
	constexpr auto rhs_is_tensor = std::is_same<T,R>::value;

	if constexpr (lhs_is_tensor && rhs_is_tensor)
		return compare(static_cast<T const&>( lhs ), static_cast<T const&>( rhs ), pred);
	else if constexpr (lhs_is_tensor && !rhs_is_tensor)
		return compare(static_cast<T const&>( lhs ), T( rhs ), pred);
	else if constexpr (!lhs_is_tensor && rhs_is_tensor)
		return compare(T( lhs ), static_cast<T const&>( rhs ), pred);
	else
		return compare(T( lhs ), T( rhs ), pred);

}
}
}


template<class T, class L, class R>
bool operator==( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l == r; } );
}
template<class T, class L, class R>
auto operator!=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l != r; } );
}
template<class T, class L, class R>
auto operator< ( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l <  r; } );
}
template<class T, class L, class R>
auto operator<=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l <= r; } );
}
template<class T, class L, class R>
auto operator> ( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l >  r; } );
}
template<class T, class L, class R>
auto operator>=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
				 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( lhs, rhs,
												   [](auto const& l, auto const& r){ return l >= r; } );
}

#endif
