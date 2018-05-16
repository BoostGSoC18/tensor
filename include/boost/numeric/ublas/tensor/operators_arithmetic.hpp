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

#ifndef _BOOST_UBLAS_TENSOR_OPERATORS_
#define _BOOST_UBLAS_TENSOR_OPERATORS_

#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <type_traits>
#include <functional>

namespace boost::numeric::ublas {
template<class element_type, class storage_format, class storage_type>
class tensor;
}


// Overloaded arithmetic operators with matrices
template<class T, class L, class R>
auto operator+( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l + r; });
}
template<class T, class L, class R>
auto operator-( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l - r; });
//	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) - rhs(i);});
}
template<class T, class L, class R>
auto operator*( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l * r; });
}
template<class T, class L, class R>
auto operator/( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs) {
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l / r; });
}

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
auto const& operator +(const boost::numeric::ublas::detail::tensor_expression<T,D>& lhs) {
	return lhs;
}

template<class T, class D>
auto operator -(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) {
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [] (auto const& l) { return -l; } );
}


#endif
