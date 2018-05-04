//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/layout.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>

BOOST_AUTO_TEST_SUITE(test_strides);


BOOST_AUTO_TEST_CASE( test_strides_ctor,
											* boost::unit_test::label("strides")
											* boost::unit_test::label("constructor")
											* boost::unit_test::depends_on("test_extents")
											* boost::unit_test::depends_on("test_layout"))
{
	using namespace boost::numeric;

	auto s1 = ublas::strides{};
	BOOST_CHECK(s1.empty());
	BOOST_CHECK_EQUAL(s1.size(), 0);

	auto s2 = ublas::strides{{1u,2u,3u}};
	BOOST_CHECK(!s2.empty());
	BOOST_CHECK_EQUAL(s2.size(), 3);

	auto e3 = ublas::extents{4,2,3};
	auto l3 = ublas::layout::first_order(e3.size());
	auto s3 = ublas::strides{e3,l3};
	BOOST_CHECK(!s3.empty());
	BOOST_CHECK_EQUAL(s3.size(), 3);

}

BOOST_AUTO_TEST_SUITE_END();
