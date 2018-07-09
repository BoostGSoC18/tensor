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


using test_types = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;




BOOST_AUTO_TEST_CASE ( test_multiplication_indices )
{
	using namespace boost::numeric::ublas::indices;


	BOOST_CHECK_EQUAL (  _a.value  ,  1  ) ;
	BOOST_CHECK_EQUAL (  _b.value  ,  2  ) ;
	BOOST_CHECK_EQUAL (  _c.value  ,  3  ) ;
	BOOST_CHECK_EQUAL (  _d.value  ,  4  ) ;
	BOOST_CHECK_EQUAL (  _e.value  ,  5  ) ;
	BOOST_CHECK_EQUAL (  _f.value  ,  6  ) ;
	BOOST_CHECK_EQUAL (  _g.value  ,  7  ) ;
	BOOST_CHECK_EQUAL (  _h.value  ,  8  ) ;
	BOOST_CHECK_EQUAL (  _i.value  ,  9  ) ;
	BOOST_CHECK_EQUAL (  _j.value  , 10  ) ;
	BOOST_CHECK_EQUAL (  _k.value  , 11  ) ;
	BOOST_CHECK_EQUAL (  _l.value  , 12  ) ;
	BOOST_CHECK_EQUAL (  _m.value  , 13  ) ;
	BOOST_CHECK_EQUAL (  _n.value  , 14  ) ;
	BOOST_CHECK_EQUAL (  _o.value  , 15  ) ;
	BOOST_CHECK_EQUAL (  _p.value  , 16  ) ;
	BOOST_CHECK_EQUAL (  _q.value  , 17  ) ;
	BOOST_CHECK_EQUAL (  _r.value  , 18  ) ;
	BOOST_CHECK_EQUAL (  _s.value  , 19  ) ;
	BOOST_CHECK_EQUAL (  _t.value  , 20  ) ;
	BOOST_CHECK_EQUAL (  _u.value  , 21  ) ;
	BOOST_CHECK_EQUAL (  _v.value  , 22  ) ;
	BOOST_CHECK_EQUAL (  _w.value  , 23  ) ;
	BOOST_CHECK_EQUAL (  _x.value  , 24  ) ;
	BOOST_CHECK_EQUAL (  _y.value  , 25  ) ;
	BOOST_CHECK_EQUAL (  _z.value  , 26  ) ;

}

BOOST_AUTO_TEST_CASE ( test_multiplication_mindices )
{
	using namespace boost::numeric::ublas;

//	BOOST_CHECK_NO_THROW(  Indices<1>(indices::c) );


	{
	MIndices<2> ind(indices::_a, indices::_b);

	BOOST_CHECK_EQUAL ( std::get<0>( ind.indices() ), 1 ) ;
	BOOST_CHECK_EQUAL ( std::get<1>( ind.indices() ), 2 ) ;

	}


	{
	MIndices<2> ind(indices::_d, indices::_c);

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

	auto t = std::make_tuple (
				indices::_a, // 0
				indices::_b, // 1
				indices::_c, // 2
				indices::_d, // 3
				indices::_e  // 4
				);

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<0>(t), std::get<2>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::_a.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::_c.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::_c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::_a.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::_c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::_d.value ) ;
	}

	{
		auto a = tensor_type(shape{2,3,4}, value_type{2});
		auto a_ind = a( std::get<2>(t), std::get<3>(t), std::get<0>(t)  );

		BOOST_CHECK_EQUAL (a_ind.first, std::addressof( a ) ) ;

		BOOST_CHECK_EQUAL (a_ind.second.at(0), indices::_c.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(1), indices::_d.value ) ;
		BOOST_CHECK_EQUAL (a_ind.second.at(2), indices::_a.value ) ;
	}

}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_einstein_multiplication, value,  test_types )
{
	using namespace boost::numeric::ublas;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = tensor<value_type,layout_type>;
	using namespace boost::numeric::ublas::indices;

	{
		auto A = tensor_type{5,3};
		auto B = tensor_type{3,4};
//		auto C = tensor_type{4,5,6};

		for(auto j = 0u; j < A.extents().at(1); ++j)
			for(auto i = 0u; i < A.extents().at(0); ++i)
				A.at( i,j ) = value_type(i+1);

		for(auto j = 0u; j < B.extents().at(1); ++j)
			for(auto i = 0u; i < B.extents().at(0); ++i)
					B.at( i,j ) = value_type(i+1);



		auto AB = A(_,_e) * B(_e,_);

//		std::cout << "A = " << A << std::endl;
//		std::cout << "B = " << B << std::endl;
//		std::cout << "AB = " << AB << std::endl;

		for(auto j = 0u; j < AB.extents().at(1); ++j)
			for(auto i = 0u; i < AB.extents().at(0); ++i)
					BOOST_CHECK_EQUAL( AB.at( i,j ) , value_type(A.at( i,0 ) * ( B.extents().at(0) * (B.extents().at(0)+1) / 2 )) );


	}


	{
		auto A = tensor_type{4,5,3};
		auto B = tensor_type{3,4,2};

		for(auto k = 0u; k < A.extents().at(2); ++k)
			for(auto j = 0u; j < A.extents().at(1); ++j)
				for(auto i = 0u; i < A.extents().at(0); ++i)
					A.at( i,j,k ) = value_type(i+1);

		for(auto k = 0u; k < B.extents().at(2); ++k)
			for(auto j = 0u; j < B.extents().at(1); ++j)
				for(auto i = 0u; i < B.extents().at(0); ++i)
					B.at( i,j,k ) = value_type(i+1);

		auto AB = A(_d,_,_f) * B(_f,_d,_);

//		std::cout << "A = " << A << std::endl;
//		std::cout << "B = " << B << std::endl;
//		std::cout << "AB = " << AB << std::endl;
		// n*(n+1)/2;
		auto const nf = ( B.extents().at(0) * (B.extents().at(0)+1) / 2 );
		auto const nd = ( A.extents().at(0) * (A.extents().at(0)+1) / 2 );

		for(auto j = 0u; j < AB.extents().at(1); ++j)
			for(auto i = 0u; i < AB.extents().at(0); ++i)
					BOOST_CHECK_EQUAL( AB.at( i,j ) ,  value_type(nf * nd) );

	}


	{
		auto A = tensor_type{4,3};
		auto B = tensor_type{3,4,2};

		for(auto j = 0u; j < A.extents().at(1); ++j)
			for(auto i = 0u; i < A.extents().at(0); ++i)
				A.at( i,j ) = value_type(i+1);

		for(auto k = 0u; k < B.extents().at(2); ++k)
			for(auto j = 0u; j < B.extents().at(1); ++j)
				for(auto i = 0u; i < B.extents().at(0); ++i)
					B.at( i,j,k ) = value_type(i+1);

		auto AB = A(_d,_f) * B(_f,_d,_);

		// n*(n+1)/2;
		auto const nf = ( B.extents().at(0) * (B.extents().at(0)+1) / 2 );
		auto const nd = ( A.extents().at(0) * (A.extents().at(0)+1) / 2 );

		for(auto i = 0u; i < AB.extents().at(0); ++i)
				BOOST_CHECK_EQUAL ( AB.at( i  ) ,  value_type(nf * nd) );

	}
}

BOOST_AUTO_TEST_SUITE_END();

