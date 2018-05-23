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
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") ) ;


using test_types = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;


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




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_vector, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type,layout_type>;
	using vector_type  = typename tensor_type::vector_type;


	for(auto const& n : extents){

		auto a = tensor_type(n, value_type{2});

		for(auto m = 0u; m < n.size(); ++m){

			auto b = vector_type  (n[m], value_type{1} );

			auto c = ublas::prod(m+1, a, b);

			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(n[m]) * a[i] );

		}
	}
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_matrix, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type,layout_type>;
	using matrix_type  = typename tensor_type::matrix_type;


	for(auto const& n : extents) {

		auto a = tensor_type(n, value_type{2});

		for(auto m = 0u; m < n.size(); ++m){

			auto b  = matrix_type  ( n[m], n[m], value_type{1} );

			auto c = ublas::prod(m+1, a, b);

			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(n[m]) * a[i] );

		}
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type,layout_type>;


	for(auto const& n : extents) {

		auto a  = tensor_type(n, value_type(2));
		auto b  = tensor_type(n, value_type(1));

		auto c = ublas::inner_prod(a, b);
		auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

		BOOST_CHECK_EQUAL( c , r );

	}
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type,layout_type>;

	for(auto const& n1 : extents) {
		auto a  = tensor_type(n1, value_type(2));
		for(auto const& n2 : extents) {

			auto b  = tensor_type(n2, value_type(1));
			auto c  = ublas::outer_prod(a, b);

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
		}
	}
}


BOOST_AUTO_TEST_SUITE_END();

