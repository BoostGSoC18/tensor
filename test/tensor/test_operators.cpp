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



#include <boost/numeric/ublas/tensor/operators.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"


using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,long,float,double,double_extended>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture {
	using extents_type = boost::numeric::ublas::basic_extents<std::size_t>;
	fixture() : extents{
				extents_type{},    // 0
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{2,3}, // 4
				extents_type{2,3,1}, // 5
				extents_type{4,1,3}, // 6
				extents_type{1,2,3}, // 7
				extents_type{4,2,3}, // 8
				extents_type{4,2,3,5} // 9
				}
	{}
	std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_entry_wise_binary_operations, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;


	auto check = [](auto const& e)
	{
		auto t  = tensor_type (e);
		auto t2 = tensor_type (e);
		auto r  = tensor_type (e);
		auto v  = value_type  {};

		std::iota(t.begin(), t.end(), v);
		std::iota(t2.begin(), t2.end(), v+2);

		r = t + t + t + t2;

//		BOOST_CHECK(  (ublas::has_tensor_types<tensor_type, decltype(tttt)>::value)  );

		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL ( r(i), 3*t(i) + t2(i) );


		r = t2 - t + t2 - t;

		for(auto i = 0ul; i < r.size(); ++i)
			BOOST_CHECK_EQUAL ( r(i), 4 );


		r = tensor_type (e,1) + tensor_type (e,1);

		for(auto i = 0ul; i < r.size(); ++i)
			BOOST_CHECK_EQUAL ( r(i), 2 );
	};

	for(auto const& e : extents)
		check(e);
}
