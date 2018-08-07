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

#ifndef _BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_
#define _BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_

#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "multi_index.hpp"
#include "functions.hpp"

#include <type_traits>
#include <functional>

namespace boost{
namespace numeric{
namespace ublas {


template<class element_type, class storage_format, class storage_type>
class tensor;

template<class E>
class matrix_expression;


template<class E>
class vector_expression;

}
}
}

#define FIRST_ORDER_OPERATOR_RIGHT(OP, EXPR_TYPE_L, EXPR_TYPE_R) \
template<class T, class L, class R> \
auto operator OP ( boost::numeric::ublas:: EXPR_TYPE_L <T,L> const& lhs, boost::numeric::ublas:: EXPR_TYPE_R <R> const& rhs) { \
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), \
	  [](auto const& l, auto const& r){ return l OP r; }); \
} \

FIRST_ORDER_OPERATOR_RIGHT (*, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (+, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (-, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (/, detail:: tensor_expression , vector_expression)

FIRST_ORDER_OPERATOR_RIGHT (*, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (+, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (-, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (/, detail:: tensor_expression , matrix_expression)


#define FIRST_ORDER_OPERATOR_LEFT(OP, EXPR_TYPE_L, EXPR_TYPE_R) \
template<class T, class L, class R> \
auto operator OP ( boost::numeric::ublas:: EXPR_TYPE_L <L> const& lhs, boost::numeric::ublas:: EXPR_TYPE_R <T,R> const& rhs) { \
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), \
	  [](auto const& l, auto const& r){ return l OP r; }); \
} \

FIRST_ORDER_OPERATOR_LEFT (*, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (+, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (-, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (/, vector_expression, detail:: tensor_expression)

FIRST_ORDER_OPERATOR_LEFT (*, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (+, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (-, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (/, matrix_expression, detail:: tensor_expression)

/** @brief Performs a tensor contraction, not an elementwise multiplication
  *
  *
  *
  * @tparam T tensor teype
  * @tparam M
*/
// this is not a elementwise multiplication, but a tensor contraction.

template<class V, class S, class A, std::size_t N, class U, class B, std::size_t M>
auto operator*(
		std::pair< boost::numeric::ublas::tensor<V,S,A> const&, boost::numeric::ublas::multi_index<N> > const& lhs,
		std::pair< boost::numeric::ublas::tensor<U,S,B> const&, boost::numeric::ublas::multi_index<M> > const& rhs)
{

	auto lhs_multi_index = lhs.second;
	auto rhs_multi_index = rhs.second;

	using vtype = std::vector<std::size_t>;

	auto pp = std::make_pair( vtype {}, vtype{}  );

	for(auto i = 0u; i < N; ++i)
		for(auto j = 0u; j < M; ++j)
			if ( lhs_multi_index.at(i) == rhs_multi_index.at(j) && lhs_multi_index.at(i) != boost::numeric::ublas::indices::_.value)
				pp.first .push_back( i+1 ),
				pp.second.push_back( j+1 );

	if(pp.first.empty())
		throw std::runtime_error("Error in boost::numeric::ublas::operator*(): number of contracting indices of lhs_multi_index is zero.");

	if(pp.first.size() != pp.second.size())
		throw std::runtime_error("Error in boost::numeric::ublas::operator*(): number of contracting indices from lhs_multi_index and rhs_multi_index must be equal.");

	return boost::numeric::ublas::prod( lhs.first, rhs.first, pp.first, pp.second );
}



template<class T, class L, class R>
auto operator+( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), [](auto const& l, auto const& r){ return l + r; });
}
template<class T, class L, class R>
auto operator-( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), [](auto const& l, auto const& r){ return l - r; });
//	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) - rhs(i);});
}
template<class T, class L, class R>
auto operator*( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), [](auto const& l, auto const& r){ return l * r; });
}
template<class T, class L, class R>
auto operator/( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), [](auto const& l, auto const& r){ return l / r; });
}


// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
auto operator+(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs(), [lhs](auto const& r){ return lhs + r; });
	//return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs + rhs(i); } );
}
template<class T, class R>
auto operator-(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs(), [lhs](auto const& r){ return lhs - r; });
}
template<class T, class R>
auto operator*(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs(), [lhs](auto const& r){ return lhs * r; });
}
template<class T, class R>
auto operator/(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs(), [lhs](auto const& r){ return lhs / r; });
}


template<class T, class L>
auto operator+(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [rhs] (auto const& l) { return l + rhs; } );
}
template<class T, class L>
auto operator-(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [rhs] (auto const& l) { return l - rhs; } );
}
template<class T, class L>
auto operator*(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [rhs] (auto const& l) { return l * rhs; } );
}
template<class T, class L>
auto operator/(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [rhs] (auto const& l) { return l / rhs; } );
}



template<class T, class D>
auto& operator += (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l+=r; } );
	return lhs;
}

template<class T, class D>
auto& operator -= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l-=r; } );
	return lhs;
}

template<class T, class D>
auto& operator *= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l*=r; } );
	return lhs;
}

template<class T, class D>
auto& operator /= (T& lhs, const boost::numeric::ublas::detail::tensor_expression<T,D> &expr) {
	boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l/=r; } );
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
auto const& operator +(const boost::numeric::ublas::detail::tensor_expression<T,D>& lhs) {
	return lhs;
}

template<class T, class D>
auto operator -(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [] (auto const& l) { return -l; } );
}


#endif
