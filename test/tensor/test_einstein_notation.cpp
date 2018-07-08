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




BOOST_AUTO_TEST_CASE ( test_multiplication_indices )
{
	using namespace boost::numeric::ublas::indices;


	BOOST_CHECK_EQUAL (  a.value  , 1  ) ;
	BOOST_CHECK_EQUAL (  b.value  , 2  ) ;
	BOOST_CHECK_EQUAL (  c.value  , 3  ) ;
	BOOST_CHECK_EQUAL (  d.value  , 4  ) ;
	BOOST_CHECK_EQUAL (  e.value  , 5  ) ;
	BOOST_CHECK_EQUAL (  f.value  , 6  ) ;

}

BOOST_AUTO_TEST_CASE ( test_multiplication_mindices )
{
	using namespace boost::numeric::ublas;

//	BOOST_CHECK_NO_THROW(  Indices<1>(indices::c) );


	{
	MIndices<2> ind(indices::a, indices::b);

	BOOST_CHECK_EQUAL ( std::get<0>( ind.indices() ), 1 ) ;
	BOOST_CHECK_EQUAL ( std::get<1>( ind.indices() ), 2 ) ;

	}


	{
	MIndices<2> ind(indices::d, indices::c);

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

	auto t = std::make_tuple (  indices::a, // 0
								indices::b, // 1
								indices::c, // 2
								indices::d, // 3
								indices::e  // 4
								);

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<0>(t), std::get<2>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::a.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::c.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::a.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::d.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3,4}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::d.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(2), indices::a.value ) ;
	}

}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_einstein_multiplication, value,  test_types )
{
	using namespace boost::numeric::ublas;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = tensor<value_type,layout_type>;

	{
		auto A = tensor_type(shape{2,3}, value_type{2});
		auto B = tensor_type(shape{3,4}, value_type{2});
		auto C = tensor_type(shape{4,5,6}, value_type{2});

		using namespace boost::numeric::ublas::indices;


		auto AB = A(d, e) * B(e, f);

	}


}

BOOST_AUTO_TEST_SUITE_END();

