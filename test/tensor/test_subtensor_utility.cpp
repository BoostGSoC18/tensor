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


#include <random>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"
#include "../../include/boost/numeric/ublas/tensor/subtensor_utility.hpp"
#include "../../include/boost/numeric/ublas/tensor/span.hpp"


BOOST_AUTO_TEST_SUITE ( subtensor_utility_testsuite ) ;



struct fixture_sliced_span {
	using span_type = boost::numeric::ublas::sliced_span;

	fixture_sliced_span()
		: spans{
				span_type(),    // 0, a(:)
				span_type(0,0), // 1, a(0:0)
				span_type(0,2), // 2, a(0:2)
				span_type(1,1), // 3, a(1:1)
				span_type(1,3),  // 4, a(1:3)
				span_type(1,boost::numeric::ublas::end), // 5, a(1:end)
				span_type(boost::numeric::ublas::end) // 6, a(end)
				}
	{}
	std::vector<span_type> spans;
};


BOOST_FIXTURE_TEST_CASE( transform_sliced_span_test, fixture_sliced_span )
{

	using namespace boost::numeric;

//	template<class size_type, class span_tag>
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(2) ) == ublas::sliced_span(0,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(3) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(4) ) == ublas::sliced_span(0,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(2) ) == ublas::sliced_span(0,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(3) ) == ublas::sliced_span(0,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(4) ) == ublas::sliced_span(0,0) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(3) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(4) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(5) ) == ublas::sliced_span(0,2) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(2) ) == ublas::sliced_span(1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(3) ) == ublas::sliced_span(1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(4) ) == ublas::sliced_span(1,1) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(4) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(5) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(6) ) == ublas::sliced_span(1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(4) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(5) ) == ublas::sliced_span(1,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(6) ) == ublas::sliced_span(1,5) );


	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(4) ) == ublas::sliced_span(3,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(5) ) == ublas::sliced_span(4,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(6) ) == ublas::sliced_span(5,5) );
}


struct fixture_strided_span {
	using span_type = boost::numeric::ublas::strided_span;

	fixture_strided_span()
		: spans{
				span_type(),       // 0, a(:)
				span_type(0,1,0),  // 1, a(0:1:0)
				span_type(0,2,2),  // 2, a(0:2:2)
				span_type(1,1,1),  // 3, a(1:1:1)
				span_type(1,1,3),  // 4, a(1:1:3)
				span_type(1,2,boost::numeric::ublas::end), // 5, a(1:2:end)
				span_type(boost::numeric::ublas::end) // 6, a(end)
				}
	{}
	std::vector<span_type> spans;
};


BOOST_FIXTURE_TEST_CASE( transform_strided_span_test, fixture_strided_span )
{

	using namespace boost::numeric;

//	template<class size_type, class span_tag>
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(2) ) == ublas::strided_span(0,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(3) ) == ublas::strided_span(0,1,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(4) ) == ublas::strided_span(0,1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(2) ) == ublas::strided_span(0,1,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(3) ) == ublas::strided_span(0,1,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(4) ) == ublas::strided_span(0,1,0) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(3) ) == ublas::strided_span(0,2,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(4) ) == ublas::strided_span(0,2,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(5) ) == ublas::strided_span(0,2,2) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(2) ) == ublas::strided_span(1,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(3) ) == ublas::strided_span(1,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(4) ) == ublas::strided_span(1,1,1) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(4) ) == ublas::strided_span(1,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(5) ) == ublas::strided_span(1,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(6) ) == ublas::strided_span(1,1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(4) ) == ublas::strided_span(1,2,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(5) ) == ublas::strided_span(1,2,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(6) ) == ublas::strided_span(1,2,5) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(4) ) == ublas::strided_span(3,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(5) ) == ublas::strided_span(4,1,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(6) ) == ublas::strided_span(5,1,5) );
}






struct fixture_shape {
	using shape = boost::numeric::ublas::shape;

	fixture_shape() : extents{
				shape{},    // 0
				shape{1,1}, // 1
				shape{1,2}, // 2
				shape{2,1}, // 3
				shape{2,3}, // 4
				shape{2,3,1}, // 5
				shape{4,1,3}, // 6
				shape{1,2,3}, // 7
				shape{4,2,3}, // 8
				shape{4,2,3,5} // 9
				}
	{}
	std::vector<shape> extents;
};

BOOST_FIXTURE_TEST_CASE( generate_span_vector_test, fixture_shape )
{
	using namespace boost::numeric::ublas;
	using span = sliced_span;

	// shape{}
	{
	auto v = detail::generate_span_vector<span>(extents[0]);
	auto r = std::vector<span>{};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}


	// shape{1,1}
	{
	auto v = detail::generate_span_vector<span>(extents[1],span(),span());
	auto r = std::vector<span>{span(0,0),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
	auto v = detail::generate_span_vector<span>(extents[1],end,span(end));
	auto r = std::vector<span>{span(0,0),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}


	{
	auto v = detail::generate_span_vector<span>(extents[1],0,end);
	auto r = std::vector<span>{span(0,0),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	// shape{1,2}
	{
	auto v = detail::generate_span_vector<span>(extents[2],0,end);
	auto r = std::vector<span>{span(0,0),span(1,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
	auto v = detail::generate_span_vector<span>(extents[2],0,1);
	auto r = std::vector<span>{span(0,0),span(1,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
	auto v = detail::generate_span_vector<span>(extents[2],span(),span());
	auto r = std::vector<span>{span(0,0),span(0,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}


}

BOOST_AUTO_TEST_SUITE_END();
