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
//namespace boost { namespace numeric { namespace ublas {
//template<class element_type, class storage_format, class storage_type>
//class tensor;
//}}}

namespace boost { namespace numeric { namespace ublas { namespace detail {

template<class T, class E>
struct has_tensor_types
{
	static constexpr bool value = false;
};


template<class T>
struct has_tensor_types<T,T>
{
	static constexpr bool value = true;
};

template<class T, class D>
struct has_tensor_types<T, tensor_expression<T,D>>
{
	static constexpr bool value = std::is_same<T,D>::value || has_tensor_types<T,D>::value;
};

}}}}

// Overloaded arithmetic operators with matrices
template<class T, class L, class R>
auto operator+(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{

	using namespace boost::numeric::ublas;

//	auto bl = std::is_same<T,L>::value;
//	auto br = std::is_same<T,R>::value;

//	auto bbl= detail::has_tensor_types<T,L>::value;
////	auto bbr= detail::has_tensor_types<T,R>::value;

//	using rhs_type = decltype(rhs);

//	auto bbBr = detail::has_tensor_types<T,rhs_type>::value;

	return boost::numeric::ublas::detail::make_binary_tensor_expression<T>
			(lhs, rhs, std::plus<typename T::value_type>());
//				[&lhs,&rhs](std::size_t i){ return lhs(i) + rhs(i);});
}
template<class T, class L, class R>
auto operator-(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) - rhs(i);});
}
template<class T, class L, class R>
auto operator*(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) * rhs(i);});
}
template<class T, class L, class R>
auto operator/(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) / rhs(i);});
}

// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
auto operator+(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type::const_reference lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs + rhs(i); } );
}
template<class T, class R>
auto operator-(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type::const_reference lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs - rhs(i); } );
}
template<class T, class R>
auto operator*(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type::const_reference lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs * rhs(i); } );
}
template<class T, class R>
auto operator/(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type::const_reference lhs,
			   boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs / rhs(i); } );
}
template<class T, class L>
auto operator+(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   typename boost::numeric::ublas::detail::tensor_expression<T,L>::tensor_type::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs(i) + rhs; } );
}
template<class T, class L>
auto operator-(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   typename boost::numeric::ublas::detail::tensor_expression<T,L>::tensor_type::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs(i) - rhs; } );
}
template<class T, class L>
auto operator*(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   typename boost::numeric::ublas::detail::tensor_expression<T,L>::tensor_type::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs(i) * rhs; } );
}
template<class T, class L>
auto operator/(boost::numeric::ublas::detail::tensor_expression<T,L> const& lhs,
			   typename boost::numeric::ublas::detail::tensor_expression<T,L>::tensor_type::const_reference rhs)
{
	return boost::numeric::ublas::detail::make_lambda<T> ( [&lhs,&rhs](std::size_t i) {return lhs(i) / rhs; } );
}




// Overloaded Assignment Operators
template<class T, class R>
decltype(auto) operator+=(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type& lhs,
						  boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{ return lhs = lhs + rhs;  }

template<class T, class R>
decltype(auto)operator-=(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type& lhs,
						 boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{ return lhs = lhs - rhs;  }

template<class T, class R>
decltype(auto) operator*=(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type& lhs,
						  boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{ return lhs = lhs * rhs;  }

template<class T, class R>
decltype(auto) operator/=(typename boost::numeric::ublas::detail::tensor_expression<T,R>::tensor_type& lhs,
						  boost::numeric::ublas::detail::tensor_expression<T,R> const& rhs)
{ return lhs = lhs / rhs;  }

// Overloaded Assignment Operators
template<class E, class F, class A>
decltype(auto) operator+=(boost::numeric::ublas::tensor<E,F,A>& lhs,
						  typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs + rhs;  }

template<class E, class F, class A>
decltype(auto) operator-=(boost::numeric::ublas::tensor<E,F,A>& lhs,
						  typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs - rhs;  }

template<class E, class F, class A>
decltype(auto) operator*=(boost::numeric::ublas::tensor<E,F,A>& lhs,
						  typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs * rhs;  }

template<class E, class F, class A>
decltype(auto) operator/=(boost::numeric::ublas::tensor<E,F,A>& lhs,
						  typename boost::numeric::ublas::tensor<E,F,A>::const_reference rhs)
{ return lhs = lhs / rhs;  }



#endif
