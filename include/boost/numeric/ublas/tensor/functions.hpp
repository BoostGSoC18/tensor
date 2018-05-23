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
#include "contraction.hpp"



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



}
}
}


#endif
