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

#ifndef _BOOST_UBLAS_TENSOR_EXPRESSIONS_
#define _BOOST_UBLAS_TENSOR_EXPRESSIONS_

#include <cstddef>
#include <boost/numeric/ublas/expression_types.hpp>

namespace boost { namespace numeric { namespace ublas {


template<class element_type, class storage_format, class storage_type>
class tensor;

template<class size_type>
class basic_extents;

//TODO: put in fwd.hpp
struct tensor_tag {};

namespace detail {

///** \brief Base class for Tensor Expression models
// *
// * it does not model the Tensor Expression concept but all derived types should.
// * The class defines a common base type and some common interface for all
// * statically derived Tensor Expression classes.
// * We implement the casts to the statically derived type.
// */

// \brief expression class for expression templates
//
// \note implements crtp - no use of virtual function calls
// 
// \tparam T element type of matrices and scalars of the expression
// \tparam D derived type that can be matrices or generic lambda functions. Must support operator()(std::size_t i)
template<class T, class D>
struct tensor_expression
		: public ublas_expression<D>
{
//	static const unsigned complexity = 0;
	using expression_type = D;
	using type_category = tensor_tag;
	using tensor_type = T;	

	BOOST_UBLAS_INLINE
	const expression_type &operator () () const { return *static_cast<const expression_type *> (this); }
	BOOST_UBLAS_INLINE
				expression_type &operator () ()       { return *static_cast<      expression_type *> (this); }

	BOOST_UBLAS_INLINE
	decltype(auto) operator()(std::size_t i) const { return static_cast<const D&>(*this)(i); }

	BOOST_UBLAS_INLINE
	decltype(auto) operator()(std::size_t i) { return static_cast<      D&>(*this)(i); }

protected :
	explicit tensor_expression() = default;
};


template<class T, class EL, class ER, class OP>
struct binary_tensor_expression
		: public tensor_expression <T, binary_tensor_expression<T,EL,ER,OP>>
{
	using self_type = binary_tensor_expression<T,EL,ER,OP>;
	using tensor_type  = T;
	using expression_type = tensor_expression <tensor_type,self_type>;
	using size_type = typename tensor_type::size_type;

	explicit binary_tensor_expression(EL const& l, ER const& r, OP o) : el(l) , er(r) , op(o) {}
	binary_tensor_expression() = delete;
	binary_tensor_expression(const binary_tensor_expression& l) = delete;
	binary_tensor_expression(binary_tensor_expression&& l) = delete;

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i) const { return op(el(i), er(i)); }

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i)       { return op(el(i), er(i)); }

	EL const& el;
	ER const& er;
	OP op;
};

// \brief helper function to simply instantiation of lambda proxy class
template<class T, class EL, class ER, class OP>
auto make_binary_tensor_expression( EL const& el, ER const& er, OP op)
{
	return binary_tensor_expression<T,EL,ER,OP>(el, er, op);
}



template<class T, class E, class OP>
struct unary_tensor_expression
		: public tensor_expression <T, unary_tensor_expression<T,E,OP>>
{

	using self_type = unary_tensor_expression<T,E,OP>;
	using tensor_type  = T;
	using expression_type = tensor_expression <tensor_type,self_type>;
	using size_type = typename tensor_type::size_type;

	explicit unary_tensor_expression(E const& ee, OP o) : e(ee) , op(o) {}
	unary_tensor_expression() = delete;
	unary_tensor_expression(const unary_tensor_expression& l) = delete;
	unary_tensor_expression(unary_tensor_expression&& l) = delete;

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i) const { return op(e(i)); }

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i)       { return op(e(i)); }

	E const& e;
	OP op;
};

// \brief helper function to simply instantiation of lambda proxy class
template<class T, class E, class OP>
auto make_unary_tensor_expression( E const& e, OP op)
{
	return unary_tensor_expression<T,E,OP>(e, op);
}



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
	static_assert(detail::has_tensor_types<T,tensor_expression<T,D>>::value, "Error in boost::numeric::ublas::tensor: Expression to evaluate should contain tensors.");

	auto const& cast_expr = static_cast<D const&>(expr);

	if constexpr ( std::is_same<T,D>::value )
		return cast_expr.extents();
	else
		return retrieve_extents(cast_expr);
}

template<class T, class EL, class ER, class OP>
auto retrieve_extents(binary_tensor_expression<T,EL,ER,OP> const& expr)
{
	static_assert(detail::has_tensor_types<T,binary_tensor_expression<T,EL,ER,OP>>::value, "Error in boost::numeric::ublas::tensor: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,EL>::value )
		return static_cast<T const&>(expr.el).extents();

	else if constexpr ( detail::has_tensor_types<T,EL>::value )
			return retrieve_extents(expr.el);

	else if constexpr ( detail::has_tensor_types<T,ER>::value  )
			return retrieve_extents(expr.er);
}


template<class T, class E, class OP>
auto retrieve_extents(unary_tensor_expression<T,E,OP> const& expr)
{

	static_assert(detail::has_tensor_types<T,unary_tensor_expression<T,E,OP>>::value, "Error in boost::numeric::ublas::tensor: Expression to evaluate should contain tensors.");


	if constexpr ( std::is_same<T,E>::value )
		return static_cast<T const&>(expr.e).extents();

	else if constexpr ( detail::has_tensor_types<T,E>::value  )
			return retrieve_extents(expr.e);
}




}}}}
#endif
