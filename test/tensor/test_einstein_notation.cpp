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

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_einstein_notation ) ; // , * boost::unit_test::depends_on("test_tensor_contraction")


//using test_types = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;




BOOST_AUTO_TEST_CASE ( test_multiplication_placeholders )
{
	using namespace boost::numeric::ublas::placeholders;


	BOOST_CHECK_EQUAL (  a.value  , 1  ) ;
	BOOST_CHECK_EQUAL (  b.value  , 2  ) ;
	BOOST_CHECK_EQUAL (  c.value  , 3  ) ;
	BOOST_CHECK_EQUAL (  d.value  , 4  ) ;
	BOOST_CHECK_EQUAL (  e.value  , 5  ) ;
	BOOST_CHECK_EQUAL (  f.value  , 6  ) ;

}

BOOST_AUTO_TEST_CASE ( test_multiplication_indices )
{
	using namespace boost::numeric::ublas;

//	BOOST_CHECK_NO_THROW(  MultiplicationIndices<1>(placeholders::c) );


	{
	MultiplicationIndices<2> ind(placeholders::a, placeholders::b);

	BOOST_CHECK_EQUAL ( std::get<0>( ind.indices() ), 1 ) ;
	BOOST_CHECK_EQUAL ( std::get<1>( ind.indices() ), 2 ) ;

	}


	{
	MultiplicationIndices<2> ind(placeholders::d, placeholders::c);

	BOOST_CHECK_EQUAL ( std::get<0>( ind.indices() ), 4 ) ;
	BOOST_CHECK_EQUAL ( std::get<1>( ind.indices() ), 3 ) ;
	}

}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_einstein_notation, value,  test_types )
{
	using namespace boost::numeric::ublas;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = tensor<value_type,layout_type>;

	auto t = std::make_tuple (  placeholders::a, // 0
								placeholders::b, // 1
								placeholders::c, // 2
								placeholders::d, // 3
								placeholders::e  // 4
								);

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<0>(t), std::get<2>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), placeholders::a.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), placeholders::c.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), placeholders::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), placeholders::a.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), placeholders::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), placeholders::d.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3,4}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), placeholders::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), placeholders::d.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(2), placeholders::a.value ) ;
	}

}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_einstein_multiplication, value,  test_types )
{
	using namespace boost::numeric::ublas;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = tensor<value_type,layout_type>;

	{
		auto A = tensor_type(shape{2,3}, value_type{2});
		auto B = tensor_type(shape{3,4}, value_type{2});
		auto C = tensor_type(shape{4,5,6}, value_type{2});

		using namespace boost::numeric::ublas::placeholders;


		auto AB = A(d, e) * B(e, f);

	}


}

BOOST_AUTO_TEST_SUITE_END();

