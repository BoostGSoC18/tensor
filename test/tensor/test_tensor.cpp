//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestTensor
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE ( test_tensor ) ;

using test_types = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
	using namespace boost::numeric;

	auto a1 = ublas::tensor<value>{};
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	auto a2 = ublas::tensor<value>{{1}};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = ublas::tensor<value>{{2}};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = ublas::tensor<value>{{1,2}};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = ublas::tensor<value>{{2,1}};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = ublas::tensor<value>{{4,3,2}};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = ublas::tensor<value>{{4,1,2}};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);
}

BOOST_AUTO_TEST_SUITE_END();
