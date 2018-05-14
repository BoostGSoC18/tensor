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


};

template<class F, class M>
class lambda;

// \brief proxy class for encapsulating generic lambdas
// 
// \tparam T element type of matrices and scalars of the expression
// \tparam F type of lambda function that is encapsulated
template<class T, class F>
class lambda
		: public tensor_expression <T, lambda<T,F>>
{
public:
	using tensor_type = T;
	using lambda_type = F;
	using expression_type = tensor_expression <tensor_type, lambda<tensor_type,lambda_type>>;
	using size_type = typename tensor_type::size_type;

	explicit lambda(lambda_type const& l)  : expression_type(), _lambda(l)  {}
	lambda() = delete;
	lambda(const lambda& l) = delete;

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i) const { return _lambda(i); }

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i)       { return _lambda(i); }
private:
	lambda_type _lambda;
};

// \brief helper function to simply instantiation of lambda proxy class
template<class T, class F>
auto make_lambda( F&& f ) { return lambda<T,F>(std::forward<F>(f)); }



template<class T, class EL, class ER, class OP>
struct binary_tensor_expression
		: public tensor_expression <T, binary_tensor_expression<T,EL,ER,OP>>
{
	using self_type = binary_tensor_expression<T,EL,ER,OP>;
	using tensor_type  = T;
	using expression_type = tensor_expression <tensor_type,self_type>;
	using size_type = typename tensor_type::size_type;

	explicit binary_tensor_expression(EL const& l, ER const& r, OP op) : el_(l) , er_(r) , op_(op) {}
	binary_tensor_expression() = delete;
	binary_tensor_expression(const binary_tensor_expression& l) = delete;

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i) const { return op_(el_(i), er_(i)); }

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i)       { return op_(el_(i), er_(i)); }


	EL const& el_;
	ER const& er_;
	OP op_;
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

	explicit unary_tensor_expression(E const& e, OP op) : e_(e) , op_(op) {}
	unary_tensor_expression() = delete;
	unary_tensor_expression(const unary_tensor_expression& l) = delete;

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i) const { return op_(e_(i)); }

	BOOST_UBLAS_INLINE
	decltype(auto)  operator()(size_type i)       { return op_(e_(i)); }

	E const& e_;
	OP op_;
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


// Lambdas cannot be supported as wanted.
template<class T, class F>
struct has_tensor_types<T, lambda<T,F>>
{ static constexpr bool value = true; };



}}}}
#endif
