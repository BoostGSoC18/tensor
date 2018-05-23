//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB in producing this work.
//
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE ( test_tensor_algorithms,
												* boost::unit_test::depends_on("test_extents")
												* boost::unit_test::depends_on("test_strides")) ;


using test_types  = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;
using test_types2 = std::tuple<int,long,float,double,std::complex<float>>; //

struct fixture {
	using extents_type = boost::numeric::ublas::shape;
	fixture() : extents {
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{2,3}, // 4
				extents_type{2,3,1}, // 5
				extents_type{4,1,3}, // 6
				extents_type{1,2,3}, // 7
				extents_type{4,2,3}, // 8
				extents_type{4,2,3,5} // 9
				}
	{}
	std::vector<extents_type> extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;


	for(auto const& n : extents) {

		auto a  = vector_type(n.product());
		auto b  = vector_type(n.product());
		auto c  = vector_type(n.product());

		auto wa = ublas::strides<ublas::first_order>(n);
		auto wb = ublas::strides<ublas::last_order> (n);
		auto wc = ublas::strides<ublas::first_order>(n);

		auto v = value_type{};
		for(auto i = 0ul; i < a.size(); ++i, v+=1){
			a[i]=v;
		}

		ublas::copy( n.size(), n.data(), b.data(), wb.data(), a.data(), wa.data() );
		ublas::copy( n.size(), n.data(), c.data(), wc.data(), b.data(), wb.data() );

		for(auto i = 1ul; i < c.size(); ++i){
			BOOST_CHECK_EQUAL( c[i], a[i] );
		}
	}
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_trans, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;




	for(auto const& n : extents) {

		auto pi = std::vector<std::size_t> (n.size());
		auto a  = vector_type(n.product());
		auto b  = vector_type(n.product());
		auto c  = vector_type(n.product());

		auto wa = ublas::strides<ublas::first_order>(n);
		auto wb = ublas::strides<ublas::last_order> (n);
		auto wc = ublas::strides<ublas::first_order>(n);

		auto v = value_type{};
		for(auto i = 0ul; i < a.size(); ++i, v+=1){
			a[i]=v;			
		}

		for(auto i = 0ul, j = n.size(); i < n.size(); ++i, --j){
			pi[i] = j;
		}

//		ublas::trans( n.size(), n.data(), pi.data(), b.data(), wb.data(), a.data(), wa.data() );
//		ublas::trans( n.size(), n.data(), pi.data(), c.data(), wc.data(), b.data(), wb.data() );

//		for(auto i = 1ul; i < c.size(); ++i){
//			BOOST_CHECK_EQUAL( c[i], a[i] );
//		}
	}
}


BOOST_AUTO_TEST_SUITE_END();

