//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE ( test_extents );


struct fixture {
	using extents_type = boost::numeric::ublas::extents;
	fixture() : extents{
								extents_type{},    // 0
								extents_type{1},   // 1
								extents_type{2},   // 2
								extents_type{1,1}, // 3
								extents_type{1,2}, // 4
								extents_type{2,1}, // 5
								extents_type{2,3}, // 6
								extents_type{2,3,1}, // 7
								extents_type{1,2,3}, // 8
								extents_type{4,2,3}}  // 9
	{}
	std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE(test_extents_ctor, fixture,
										 *boost::unit_test::label("extents")
										 *boost::unit_test::label("constructor"))
{
	using namespace boost::numeric;

	BOOST_REQUIRE_EQUAL(extents.size(),10);

	BOOST_CHECK( extents[0].empty());
	for(auto i = 1u; i < extents.size(); ++i)
		BOOST_CHECK(!extents[i].empty());

	BOOST_CHECK_EQUAL(extents[0].size(), 0);

	for(auto i = 1u; i < 3u; ++i)
		BOOST_CHECK_EQUAL(extents[i].size(), 1);

	for(auto i = 3u; i < 7u; ++i)
		BOOST_CHECK_EQUAL(extents[i].size(), 2);

	for(auto i = 7u; i < 9u; ++i)
		BOOST_CHECK_EQUAL(extents[i].size(), 3);



	BOOST_CHECK_THROW( ublas::extents({1,0}), std::length_error );
	BOOST_CHECK_THROW( ublas::extents({0}  ), std::length_error );
	BOOST_CHECK_THROW( ublas::extents({0,1}), std::length_error );
}

BOOST_FIXTURE_TEST_CASE(test_extents_access, fixture,
										 *boost::unit_test::label("extents")
										 *boost::unit_test::label("access"))
{
	using namespace boost::numeric;
	BOOST_REQUIRE_EQUAL(extents.size(),10);

	BOOST_CHECK_EQUAL(extents[0].size(), 0);

	BOOST_REQUIRE_EQUAL(extents[1].size(), 1);
	BOOST_REQUIRE_EQUAL(extents[2].size(), 1);

	BOOST_CHECK_EQUAL (extents[1][0],1);
	BOOST_CHECK_EQUAL (extents[2][0],2);

	BOOST_REQUIRE_EQUAL(extents[3].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[4].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[5].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[6].size(), 2);

	BOOST_CHECK_EQUAL(extents[3][0],1);
	BOOST_CHECK_EQUAL(extents[3][1],1);

	BOOST_CHECK_EQUAL(extents[4][0],1);
	BOOST_CHECK_EQUAL(extents[4][1],2);

	BOOST_CHECK_EQUAL(extents[5][0],2);
	BOOST_CHECK_EQUAL(extents[5][1],1);

	BOOST_CHECK_EQUAL(extents[6][0],2);
	BOOST_CHECK_EQUAL(extents[6][1],3);


	BOOST_REQUIRE_EQUAL(extents[7].size(), 3);
	BOOST_REQUIRE_EQUAL(extents[8].size(), 3);
	BOOST_REQUIRE_EQUAL(extents[9].size(), 3);

	BOOST_CHECK_EQUAL(extents[7][0],2);
	BOOST_CHECK_EQUAL(extents[7][1],3);
	BOOST_CHECK_EQUAL(extents[7][2],1);

	BOOST_CHECK_EQUAL(extents[8][0],1);
	BOOST_CHECK_EQUAL(extents[8][1],2);
	BOOST_CHECK_EQUAL(extents[8][2],3);

	BOOST_CHECK_EQUAL(extents[9][0],4);
	BOOST_CHECK_EQUAL(extents[9][1],2);
	BOOST_CHECK_EQUAL(extents[9][2],3);

	BOOST_CHECK_NO_THROW( extents[6][2] );
	BOOST_CHECK_NO_THROW( extents[9][3] );
}

BOOST_AUTO_TEST_SUITE_END();

