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
bool compare(tensor<T,F,A> const& lhs, tensor<T,F,A> const& rhs, BinaryPred pred)
{
	for(auto i = 0u; i < lhs.size(); ++i)
		if(!pred(lhs(i), rhs(i)))
			return false;
	return true;
}
}
}


template<class T, class L, class R>
bool operator==( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l == r; } );
}
template<class T, class L, class R>
auto operator!=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l != r; } );
}
template<class T, class L, class R>
auto operator< ( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l <  r; } );
}
template<class T, class L, class R>
auto operator<=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l <= r; } );
}
template<class T, class L, class R>
auto operator> ( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l >  r; } );
}
template<class T, class L, class R>
auto operator>=( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::compare( T(lhs), T(rhs),  [](auto const& l, auto const& r){ return l >= r; } );
}



#if 0
// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
auto operator+(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs + r; });
	//return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs + rhs(i); } );
}
template<class T, class R>
auto operator-(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs - r; });
}
template<class T, class R>
auto operator*(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs * r; });
}
template<class T, class R>
auto operator/(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs / r; });
}


template<class T, class L>
auto operator+(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l + rhs; } );
}
template<class T, class L>
auto operator-(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l - rhs; } );
}
template<class T, class L>
auto operator*(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l * rhs; } );
}
template<class T, class L>
auto operator/(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l / rhs; } );
}



template<class T, class D>
auto& operator += (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr, [](auto& l, auto const& r) { l+=r; } );
	return lhs;
}

template<class T, class D>
auto& operator -= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr, [](auto& l, auto const& r) { l-=r; } );
	return lhs;
}

template<class T, class D>
auto& operator *= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr, [](auto& l, auto const& r) { l*=r; } );
	return lhs;
}

template<class T, class D>
auto& operator /= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr, [](auto& l, auto const& r) { l/=r; } );
	return lhs;
}



template<class E, class F, class A>
auto& operator += (boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference r) {
	boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l+=r; } );
	return lhs;
}

template<class E, class F, class A>
auto& operator -= (boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference r) {
	boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l-=r; } );
	return lhs;
}

template<class E, class F, class A>
auto& operator *= (boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference r) {
	boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l*=r; } );
	return lhs;
}

template<class E, class F, class A>
auto& operator /= (boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference r) {
	boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l/=r; } );
	return lhs;
}

template<class T, class D>
auto operator !(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [] (auto const& l) { return -l; } );
}

#endif

#endif
