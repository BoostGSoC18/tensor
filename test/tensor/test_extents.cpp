//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>

BOOST_AUTO_TEST_SUITE ( test_extents );

BOOST_AUTO_TEST_CASE(test_extents_ctor,
										 *boost::unit_test::label("extents")
										 *boost::unit_test::label("constructor"))
{
	using namespace boost::numeric;

	ublas::extents l;
	BOOST_CHECK(l.empty());
	BOOST_CHECK_EQUAL(l.size(), 0);

	auto l1 = ublas::extents{2,2};

	BOOST_REQUIRE_EQUAL(l1.size(), 2);
	BOOST_CHECK_EQUAL(l1[0], 2);
	BOOST_CHECK_EQUAL(l1[1], 2);

	auto l2 = ublas::extents{1,1};

	BOOST_REQUIRE_EQUAL(l2.size(), 2);
	BOOST_CHECK_EQUAL(l2[0], 1);
	BOOST_CHECK_EQUAL(l2[1], 1);

	auto l3 = ublas::extents{1,2};

	BOOST_REQUIRE_EQUAL(l3.size(), 2);
	BOOST_CHECK_EQUAL(l3[0], 1);
	BOOST_CHECK_EQUAL(l3[1], 2);

	auto l4 = ublas::extents{2,1};

	BOOST_REQUIRE_EQUAL(l4.size(), 2);
	BOOST_CHECK_EQUAL(l4[0], 2);
	BOOST_CHECK_EQUAL(l4[1], 1);

	auto l5 = ublas::extents{2,2};

	BOOST_REQUIRE_EQUAL(l5.size(), 2);
	BOOST_CHECK_EQUAL(l5[0], 2);
	BOOST_CHECK_EQUAL(l5[1], 2);

	auto l6 = ublas::extents{2,2,1};

	BOOST_REQUIRE_EQUAL(l6.size(), 3);
	BOOST_CHECK_EQUAL(l6[0], 2);
	BOOST_CHECK_EQUAL(l6[1], 2);
	BOOST_CHECK_EQUAL(l6[2], 1);

	auto l7 = ublas::extents{2,2,3};

	BOOST_REQUIRE_EQUAL(l7.size(), 3);
	BOOST_CHECK_EQUAL(l7[0], 2);
	BOOST_CHECK_EQUAL(l7[1], 2);
	BOOST_CHECK_EQUAL(l7[2], 3);

	auto l8 = ublas::extents{2,1,3,1};

	BOOST_REQUIRE_EQUAL(l8.size(), 4);
	BOOST_CHECK_EQUAL(l8[0], 2);
	BOOST_CHECK_EQUAL(l8[1], 1);
	BOOST_CHECK_EQUAL(l8[2], 3);

	auto l9 = ublas::extents{2,1,3,1,4};

	BOOST_REQUIRE_EQUAL(l9.size(), 5);
	BOOST_CHECK_EQUAL(l9[0], 2);
	BOOST_CHECK_EQUAL(l9[1], 1);
	BOOST_CHECK_EQUAL(l9[2], 3);
	BOOST_CHECK_EQUAL(l9[3], 1);
	BOOST_CHECK_EQUAL(l9[4], 4);
}

BOOST_AUTO_TEST_SUITE_END();

