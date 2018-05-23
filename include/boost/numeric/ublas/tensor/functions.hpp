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


#ifndef _BOOST_UBLAS_TENSOR_FUNCTIONS_
#define _BOOST_UBLAS_TENSOR_FUNCTIONS_


#include <stdexcept>
#include <vector>
#include "multiplication.hpp"



namespace boost {
namespace numeric {
namespace ublas {

template<class Value, class Format, class Allocator>
class tensor;

template<class Value, class Format, class Allocator>
class matrix;

template<class Value, class Allocator>
class vector;





template<class V, class F, class A1, class A2>
auto prod(const std::size_t m, tensor<V,F,A1> const& a, vector<V,A2> const& b)
{

	using tensor_type  = tensor<V,F,A1>;
	using extents_type = typename tensor_type::extents_type;
	using ebase_type   = typename extents_type::base_type;
	using value_type   = typename tensor_type::value_type;

	auto const p = a.rank();

	if( m == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Contraction mode must be greater than zero.");

	if( p < m )
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater equal the modus.");

	if( p == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater than zero.");

	if( a.empty() )
		throw std::length_error("Error in boost::numeric::ublas::prod: tensor should not be empty.");

	if( b.size() == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: vector should not be empty.");


	auto nc = ebase_type(std::max(p-1,2ul) ,1);
	auto nb = ebase_type{b.size(),1};


	for(auto i = 0u, j = 0u; i < p; ++i)
		if(i != m-1)
			nc[j++] = a.extents().at(i);

	auto c = tensor_type(extents_type(nc),value_type{});

	auto bb = &(b(0));

	ttv(m, p,
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			bb, nb.data(), nb.data());


	return c;
}


template<class V, class F, class A1, class A2>
auto prod(const std::size_t m, tensor<V,F,A1> const& a, matrix<V,F,A2> const& b)
{

	using tensor_type  = tensor<V,F,A1>;
	using extents_type = typename tensor_type::extents_type;
	using strides_type = typename tensor_type::strides_type;
	using value_type   = typename tensor_type::value_type;


	auto const p = a.rank();

	if( m == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Contraction mode must be greater than zero.");

	if( p < m || m > a.extents().size())
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater equal the modus.");

	if( p == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater than zero.");

	if( a.empty() )
		throw std::length_error("Error in boost::numeric::ublas::prod: tensor should not be empty.");

	if( b.size1()*b.size2() == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: matrix should not be empty.");


	auto nc = a.extents().base();
	auto nb = extents_type {b.size1(),b.size2()};
	auto wb = strides_type (nb);

	nc[m-1] = nb[0];

	auto c = tensor_type(extents_type(nc),value_type{});

	auto bb = &(b(0,0));

	ttm(m, p,
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			bb, nb.data(), wb.data());


	return c;
}


/** @brief Computes the inner product of two tensors
 *
 * Implements c = sum(A[i1,i2,...,ip] * B[i1,i2,...,jp])
 *
 * @note calls inner function
 *
 * @param[in] a tensor object A
 * @param[in] b tensor object B
 *
 * @returns a value type.
*/
template<class V, class F, class A1, class A2>
auto inner_prod(tensor<V,F,A1> const& a, tensor<V,F,A2> const& b)
{
	using value_type   = typename tensor<V,F,A1>::value_type;

	if( a.rank() != b.rank() )
		throw std::length_error("Error in boost::numeric::ublas::inner_prod: Rank of both tensors must be the same.");

	if( a.empty() || b.empty())
		throw std::length_error("Error in boost::numeric::ublas::inner_prod: Tensors should not be empty.");

	if( a.extents() != b.extents())
		throw std::length_error("Error in boost::numeric::ublas::inner_prod: Tensor extents should be the same.");

	return inner(a.rank(), a.extents().data(),
							 a.data(), a.strides().data(),
							 b.data(), b.strides().data(), value_type{0});
}

/** @brief Computes the outer product of two tensors
 *
 * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
 *
 * @note calls outer function
 *
 * @param[in] a tensor object A
 * @param[in] b tensor object B
 *
 * @returns tensor object C with the same storage format F and allocator type A1
*/
template<class V, class F, class A1, class A2>
auto outer_prod(tensor<V,F,A1> const& a, tensor<V,F,A2> const& b)
{
	using tensor_type  = tensor<V,F,A1>;
	using extents_type = typename tensor_type::extents_type;

	if( a.empty() || b.empty() )
		throw std::length_error("Error in boost::numeric::ublas::outer_prod: tensors should not be empty.");

	auto nc = typename extents_type::base_type(a.rank() + b.rank());
	for(auto i = 0u; i < a.rank(); ++i)
		nc.at(i) = a.extents().at(i);

	for(auto i = 0u; i < b.rank(); ++i)
		nc.at(a.rank()+i) = b.extents().at(i);

	auto c = tensor_type(extents_type(nc));

	outer(c.data(), c.rank(), c.extents().data(), c.strides().data(),
				a.data(), a.rank(), a.extents().data(), a.strides().data(),
				b.data(), b.rank(), b.extents().data(), b.strides().data());

	return c;
}



/** @brief Transposes a tensor according to a permutation tuple
 *
 * Implements C[tau[i1],tau[i2]...,tau[ip]] = A[i1,i2,...,ip]
 *
 * @note calls trans function
 *
 * @param[in] a    tensor object of rank p
 * @param[in] tau  one-based permutation tuple of length p
 * @returns        a transposed tensor object with the same storage format F and allocator type A
*/
template<class V, class F, class A>
auto trans(tensor<V,F,A> const& a, std::vector<std::size_t> const& tau)
{
	using tensor_type  = tensor<V,F,A>;
	using extents_type = typename tensor_type::extents_type;

	if( a.empty() )
		throw std::length_error("Error in boost::numeric::ublas::trans: tensor should not be empty.");

	auto const   p = a.rank();
	auto const& na = a.extents();

	auto nc = typename extents_type::base_type(p);
	for(auto i = 0u; i < p; ++i)
		nc[i] = na[tau[i]-1];

	auto c = tensor_type(extents_type(nc));

//	SizeType const p,  SizeType const*const n, SizeType const*const pi,
//							PointerOut c,      SizeType const*const wc,
//							const PointerIn a, SizeType const*const wa)


	trans(a.rank(), a.extents().data(), tau.data(), c.data(), c.strides().data(), a.data(), a.strides().data() );



//	auto vc = typename strides_type::base_type(p);
//	for(size_t i = 0u; i < p; ++i)
//		vc[tau[i]-1] = wc[i];



//	auto c = tensor<V,F,A>(extents_type(nc));

//	outer(c.data(), c.rank(), c.extents().data(), c.strides().data(),
//				a.data(), a.rank(), a.extents().data(), a.strides().data(),
//				b.data(), b.rank(), b.extents().data(), b.strides().data());

	return c;
}


}
}
}


#endif
