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
#include <type_traits>
#include <functional>


// Overloaded arithmetic operators with matrices
template<class T, class L, class R>
auto operator+( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l + r; });
}
template<class T, class L, class R>
auto operator-( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l - r; });
//	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) - rhs(i);});
}
template<class T, class L, class R>
auto operator*( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l * r; });
}
template<class T, class L, class R>
auto operator/( boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs, rhs, [](auto const& l, auto const& r){ return l / r; });
}

// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
auto operator+(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{	
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs + r; });
	//return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs + rhs(i); } );
}
template<class T, class R>
auto operator-(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs - r; });
}
template<class T, class R>
auto operator*(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs * r; });
}
template<class T, class R>
auto operator/(typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (rhs, [lhs](auto const& r){ return lhs / r; });
}


template<class T, class L>
auto operator+(
		boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs)
{	
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l + rhs; } );
}
template<class T, class L>
auto operator-(
		boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l - rhs; } );
}
template<class T, class L>
auto operator*(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l * rhs; } );
}
template<class T, class L>
auto operator/(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs, typename T::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs, [rhs] (auto const& l) { return l / rhs; } );
}




// Overloaded Assignment Operators
template<class E, class F, class A>
decltype(auto) operator+=(boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs + rhs;  }

template<class E, class F, class A>
decltype(auto) operator-=(boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs - rhs;  }

template<class E, class F, class A>
decltype(auto) operator*=(boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs * rhs;  }

template<class E, class F, class A>
decltype(auto) operator/=(boost::numeric::ublas::tensor<E,F,A>& lhs, typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs / rhs;  }



#endif
