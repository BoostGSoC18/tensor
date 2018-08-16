//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
#include <boost/test/unit_test.hpp>


#include "utility.hpp"
#include "../../include/boost/numeric/ublas/tensor/tensor.hpp"
#include "../../include/boost/numeric/ublas/tensor/algorithms_traits.hpp"



BOOST_AUTO_TEST_SUITE ( algorithms_traits_testsuite ) ;


using test_types  = zip<int,long>::with_t<boost::numeric::ublas::tag::first_order, boost::numeric::ublas::tag::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( comparison_test, value, test_types )
{
	using namespace boost::numeric;
	using value_type = typename value::first_type;
	using layout_type = typename value::second_type;
	using trait = ublas::detail::recursive_function_traits<value_type,layout_type,ublas::detail::tag::unit_access>;
	using compare_type = typename trait::compare_type;


	auto comp = compare_type{4};

	// comp = r>0
	if( std::is_same_v<layout_type, ublas::tag::first_order>){

		BOOST_CHECK( !comp(0) );
		BOOST_CHECK(  comp(1) );
		BOOST_CHECK(  comp(2) );
		BOOST_CHECK(  comp(3) );
		BOOST_CHECK(  comp(4) );
		BOOST_CHECK(  comp(5) );
		BOOST_CHECK(  comp(6) );
	}

	// comp = r<4
	if( std::is_same_v<layout_type, ublas::tag::last_order>){
		BOOST_CHECK(  comp(0) );
		BOOST_CHECK(  comp(1) );
		BOOST_CHECK(  comp(2) );
		BOOST_CHECK(  comp(3) );
		BOOST_CHECK( !comp(4) );
		BOOST_CHECK( !comp(5) );
		BOOST_CHECK( !comp(6) );
	}
}




BOOST_AUTO_TEST_CASE_TEMPLATE( level_increment_test, value, test_types )
{
	using namespace boost::numeric;
	using value_type = typename value::first_type;
	using layout_type = typename value::second_type;
	using trait = ublas::detail::recursive_function_traits<value_type,layout_type,ublas::detail::tag::unit_access>;
	using increment_type = typename trait::increment_recursion_type;


	auto incr = increment_type{};

	// comp = r>0
	if( std::is_same_v<layout_type, ublas::tag::first_order>){
		BOOST_CHECK_EQUAL( incr(1),0 );
		BOOST_CHECK_EQUAL( incr(2),1 );
		BOOST_CHECK_EQUAL( incr(3),2 );
		BOOST_CHECK_EQUAL( incr(4),3 );
		BOOST_CHECK_EQUAL( incr(5),4 );
		BOOST_CHECK_EQUAL( incr(6),5 );
		BOOST_CHECK_EQUAL( incr(7),6 );
	}

	// comp = r<4
	if( std::is_same_v<layout_type, ublas::tag::last_order>){
		BOOST_CHECK_EQUAL( incr(0),1 );
		BOOST_CHECK_EQUAL( incr(1),2 );
		BOOST_CHECK_EQUAL( incr(2),3 );
		BOOST_CHECK_EQUAL( incr(3),4 );
		BOOST_CHECK_EQUAL( incr(4),5 );
		BOOST_CHECK_EQUAL( incr(5),6 );
		BOOST_CHECK_EQUAL( incr(6),7 );
	}
}



BOOST_AUTO_TEST_CASE_TEMPLATE( pointer_incr_test, value, test_types )
{
	using namespace boost::numeric;
	using value_type = typename value::first_type;
	using layout_type = typename value::second_type;
	using trait = ublas::detail::recursive_function_traits<value_type,layout_type,ublas::detail::tag::unit_access>;
	using increment_type = typename trait::increment_pointer_type;
	using size_type = std::size_t;

	constexpr size_type n = 10u;
	constexpr size_type p = 2u;
	size_type r = 0;
	value_type a[n], b[n];
	size_type wa[p], wb[p];

	auto incr = increment_type{p};

	std::fill(a, a+n, value_type{});
	std::fill(b, b+n, value_type{});

	std::fill(wa, wa+p, value_type{1});
	std::fill(wb, wb+p, value_type{1});

	for(size_type i = 0u; i < n; ++i){
		*a = value_type(2), *b = value_type(2);
		incr(a,wa);
		incr(b,wb);
	}

	for(size_type i = 0u; i < n; ++i){
		BOOST_CHECK_EQUAL(*a,value_type(2));
		incr(a,wa);
		incr(b,wb);
	}
}


BOOST_AUTO_TEST_SUITE_END();
