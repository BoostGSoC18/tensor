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
// \note implements crtp - use of virtual function calls
// 
// \tparam T element type of matrices and scalars of the expression
// \tparam D derived type that can be matrices or generic lambda functions. Must support operator()(std::size_t i)
template<class T, class D>
struct tensor_expression
		: public ublas_expression<D>
{
	static const unsigned complexity = 0;
	using expression_type = D;
	using type_category = tensor_tag;
	using tensor_type = T;
//	using size_type = typename tensor_type::size_type;


	/* E can be an incomplete type - to define the following we would need more template arguments
	typedef typename E::size_type size_type;
	*/

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
class lambda : public tensor_expression <T, lambda<T,F>>
{
public:
	using tensor_type = T;
	using lambda_type = F;
	using expression_type = tensor_expression <tensor_type, lambda<tensor_type,lambda_type>>;
	using size_type = typename tensor_type::size_type;

	explicit lambda(lambda_type const& l)  : expression_type{}, _lambda { l }  {}
	decltype(auto)
	operator()(size_type i) const { return _lambda(i); }
	decltype(auto)
	operator()(size_type i)       { return _lambda(i); }
private:
	lambda_type _lambda;
};
// \brief helper function to simply instantiation of lambda proxy class 
template<class F, class T>
auto make_lambda( F&& f ) { return lambda<T,F>(std::forward<F>(f)); }
}
}}}
#endif
