//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)



#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/layout.hpp>

BOOST_AUTO_TEST_SUITE(test_layout);

BOOST_AUTO_TEST_CASE(test_layout_ctor,
											*boost::unit_test::label("layout")
											*boost::unit_test::label("constructor"))
{
	using namespace boost::numeric;

	auto l1 = ublas::layout{};
	BOOST_CHECK(l1.empty());
	BOOST_CHECK_EQUAL(l1.size(), 0);

	auto l2 = ublas::layout{{1u,2u,3u}};
	BOOST_CHECK(!l2.empty());
	BOOST_CHECK_EQUAL(l2.size(), 3);

	auto l3 = ublas::layout{{1,2,3}};
	BOOST_CHECK(!l3.empty());
	BOOST_CHECK_EQUAL(l3.size(), 3);

	BOOST_CHECK(!ublas::layout({1,2,3,4}).empty());
	BOOST_CHECK_EQUAL(ublas::layout({3,2,1,4}).size(), 4);

	auto l4 = l3;
	BOOST_REQUIRE(!l4.empty());
	BOOST_REQUIRE_EQUAL(l4.size(), 3);
	BOOST_CHECK_EQUAL(l4[0],1);
	BOOST_CHECK_EQUAL(l4[1],2);
	BOOST_CHECK_EQUAL(l4[2],3);

	auto l5 = ublas::layout{3,2,1};
	BOOST_REQUIRE(!l5.empty());
	BOOST_REQUIRE_EQUAL(l5.size(), 3);
	BOOST_CHECK_EQUAL(l5[0],3);
	BOOST_CHECK_EQUAL(l5[1],2);
	BOOST_CHECK_EQUAL(l5[2],1);

	auto l7 = ublas::layout::first_order(3);
	BOOST_CHECK(l7 == l3);

	auto l8 = ublas::layout::last_order(3);
	BOOST_CHECK(l8 == l5);
	BOOST_CHECK(l8 != l3);

	auto l9 = ublas::layout{1,2,3};
	BOOST_CHECK(l9 == l3);

	BOOST_CHECK( (ublas::layout{1,2,3} == l3) );
}

BOOST_AUTO_TEST_SUITE_END();
