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

#ifndef _BOOST_UBLAS_TENSOR_EXPRESSIONS_EVALUATION_
#define _BOOST_UBLAS_TENSOR_EXPRESSIONS_EVALUATION_

#include <type_traits>


namespace boost::numeric::ublas {

template<class element_type, class storage_format, class storage_type>
class tensor;

template<class size_type>
class basic_extents;

template<class T, class D>
struct tensor_expression;

template<class T, class EL, class ER, class OP>
struct binary_tensor_expression;

template<class T, class E, class OP>
struct unary_tensor_expression;

}

namespace boost::numeric::ublas::detail {

template<class T, class E>
struct has_tensor_types
{ static constexpr bool value = false; };

template<class T>
struct has_tensor_types<T,T>
{ static constexpr bool value = true; };

template<class T, class D>
struct has_tensor_types<T, tensor_expression<T,D>>
{ static constexpr bool value = std::is_same<T,D>::value || has_tensor_types<T,D>::value; };


template<class T, class EL, class ER, class OP>
struct has_tensor_types<T, binary_tensor_expression<T,EL,ER,OP>>
{ static constexpr bool value = std::is_same<T,EL>::value || std::is_same<T,ER>::value || has_tensor_types<T,EL>::value || has_tensor_types<T,ER>::value;  };

template<class T, class E, class OP>
struct has_tensor_types<T, unary_tensor_expression<T,E,OP>>
{ static constexpr bool value = std::is_same<T,E>::value || has_tensor_types<T,E>::value; };





template<class T, class F, class A>
auto retrieve_extents(tensor<T,F,A> const& t)
{
	return t.extents();
}

template<class T, class D>
auto retrieve_extents(tensor_expression<T,D> const& expr)
{
	static_assert(detail::has_tensor_types<T,tensor_expression<T,D>>::value,
								"Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& cast_expr = static_cast<D const&>(expr);

	if constexpr ( std::is_same<T,D>::value )
		return cast_expr.extents();
	else
		return retrieve_extents(cast_expr);
}

template<class T, class EL, class ER, class OP>
auto retrieve_extents(binary_tensor_expression<T,EL,ER,OP> const& expr)
{
	static_assert(detail::has_tensor_types<T,binary_tensor_expression<T,EL,ER,OP>>::value,
								"Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,EL>::value )
		return expr.derived_left().extents();

	if constexpr ( std::is_same<T,ER>::value )
		return expr.derived_right().extents();

	else if constexpr ( detail::has_tensor_types<T,EL>::value )
			return retrieve_extents(expr.el);

	else if constexpr ( detail::has_tensor_types<T,ER>::value  )
			return retrieve_extents(expr.er);
}


template<class T, class E, class OP>
auto retrieve_extents(unary_tensor_expression<T,E,OP> const& expr)
{

	static_assert(detail::has_tensor_types<T,unary_tensor_expression<T,E,OP>>::value,
								"Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,E>::value )
		return expr.derived().extents();

	else if constexpr ( detail::has_tensor_types<T,E>::value  )
			return retrieve_extents(expr.e);
}


///////////////


template<class T, class F, class A, class S>
auto all_extents_equal(tensor<T,F,A> const& t, basic_extents<S> const& extents, bool all_equal = true)
{
	return all_equal && extents == t.extents();
}

template<class T, class D, class S>
auto all_extents_equal(tensor_expression<T,D> const& expr, basic_extents<S> const& extents, bool all_equal = true)
{
	static_assert(detail::has_tensor_types<T,tensor_expression<T,D>>::value,
								"Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");
	auto const& cast_expr = static_cast<D const&>(expr);

	if(!all_equal)
		return false;

	if constexpr ( std::is_same<T,D>::value )
		if( extents != cast_expr.extents() )
			return false;

	if constexpr ( detail::has_tensor_types<T,D>::value )
		if ( !all_extents_equal(cast_expr, extents, all_equal))
			return false;

	return true;

}

template<class T, class EL, class ER, class OP, class S>
auto all_extents_equal(binary_tensor_expression<T,EL,ER,OP> const& expr, basic_extents<S> const& extents, bool all_equal = true)
{
	static_assert(detail::has_tensor_types<T,binary_tensor_expression<T,EL,ER,OP>>::value,
								"Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");

	if(!all_equal)
		return false;

	if constexpr ( std::is_same<T,EL>::value )
		if(extents != expr.derived_left().extents())
			return false;

	if constexpr ( std::is_same<T,ER>::value )
		if(extents != expr.derived_right().extents())
			return false;

	if constexpr ( detail::has_tensor_types<T,EL>::value )
		if(!all_extents_equal(expr.el, extents, all_equal))
			return false;

	if constexpr ( detail::has_tensor_types<T,ER>::value )
		if(!all_extents_equal(expr.er, extents, all_equal))
			return false;

	return true;
}


template<class T, class E, class OP, class S>
auto all_extents_equal(unary_tensor_expression<T,E,OP> const& expr, basic_extents<S> const& extents, bool all_equal = true)
{

	static_assert(detail::has_tensor_types<T,unary_tensor_expression<T,E,OP>>::value,
								"Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");

	if(!all_equal)
		return false;

	if constexpr ( std::is_same<T,E>::value )
		if(extents != expr.derived().extents())
			return false;

	if constexpr ( detail::has_tensor_types<T,E>::value )
		if(!all_extents_equal(expr.e, extents, all_equal))
			return false;

	return true;
}




/** @brief Evaluates expression for a tensor
 *
 * Assigns the results of the expression to the tensor.
 *
 * \note Checks if shape of the tensor matches those of all tensors within the expression.
*/
template<class tensor_type, class derived_type>
void eval(tensor_type& lhs, tensor_expression<tensor_type, derived_type> const& expr)
{
	if constexpr (detail::has_tensor_types<tensor_type, tensor_expression<tensor_type,derived_type> >::value )
		if(!detail::all_extents_equal(expr, lhs.extents() ))
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: expression contains tensors with different shapes.");

	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		lhs(i) = expr(i);
}

/** @brief Evaluates expression for a tensor
 *
 * Applies a unary function to the results of the expressions before the assignment.
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor matches those of all tensors within the expression.
*/
template<class tensor_type, class derived_type, class unary_fn>
void eval(tensor_type& lhs, tensor_expression<tensor_type, derived_type> const& expr, unary_fn const fn)
{

	if constexpr (detail::has_tensor_types< tensor_type, tensor_expression<tensor_type,derived_type> >::value )
		if(!detail::all_extents_equal( expr, lhs.extents() ))
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: expression contains tensors with different shapes.");

	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		fn(lhs(i), expr(i));
}



/** @brief Evaluates expression for a tensor
 *
 * Applies a unary function to the results of the expressions before the assignment.
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor matches those of all tensors within the expression.
*/
template<class tensor_type, class unary_fn>
void eval(tensor_type& lhs, unary_fn const fn)
{
	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		fn(lhs(i));
}


}
#endif
