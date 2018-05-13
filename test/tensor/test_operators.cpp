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


using double_extended = typename boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,long,float,double,double_extended>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;



BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_entry_wise_operations, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto t = tensor_type{5,4,3};
	auto v = value_type{};
	std::iota(t.begin(), t.end(), v);

	tensor_type r(t.extents());
	r = t + t + t;

	for(auto i = 0ul; i < t.size(); ++i , ++v)
		BOOST_CHECK_EQUAL ( r(i), 3*v );


}
