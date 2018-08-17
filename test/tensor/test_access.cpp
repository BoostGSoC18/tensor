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
#include <boost/test/unit_test.hpp>


#include "../../include/boost/numeric/ublas/tensor/access.hpp"
#include "../../include/boost/numeric/ublas/tensor/extents.hpp"
#include "../../include/boost/numeric/ublas/tensor/strides.hpp"
#include "utility.hpp"



//, * boost::unit_test::depends_on("strides_testsuite")
BOOST_AUTO_TEST_SUITE ( access_testsuite ) ;


using test_types  = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::tag::first_order, boost::numeric::ublas::tag::last_order>;

struct fixture {
	using extents_type = boost::numeric::ublas::shape;
	fixture() : extents {
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

BOOST_FIXTURE_TEST_CASE_TEMPLATE( access_test, value,  test_types, fixture)
{
	using namespace boost::numeric;
//	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using shape_type   = ublas::shape;
	using size_type    = typename shape_type::value_type;
	using strides_type = ublas::strides<layout_type>;

	auto check1 = [](const shape_type& e, const strides_type& w)
	{
		auto v = size_type{};
		constexpr auto zero = size_type{0};
		for(auto k = 0ul; k < e.product(); ++k, ++v){
			BOOST_CHECK_EQUAL(ublas::detail::access<zero>(zero, w, k), v);
		}
	};

	auto check2 = [](const shape_type& e, const strides_type& w)
	{
		std::array<unsigned,2> k;
		auto r = std::is_same_v<layout_type,ublas::tag::first_order> ? 1 : 0;
		auto q = std::is_same_v<layout_type,ublas::tag::last_order > ? 1 : 0;
		auto v = size_type{};
		constexpr auto zero = size_type{0};
		for(k[r] = 0ul; k[r] < e.at(r); ++k[r])
			for(k[q] = 0ul; k[q] < e.at(q); ++k[q], ++v)
				BOOST_CHECK_EQUAL(ublas::detail::access<zero>(zero, w, k[0], k[1]), v);
	};

	auto check3 = [](const shape_type& e, const strides_type& w)
	{
		std::array<unsigned,3> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::tag::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::tag::first_order> ? 2 : 0;
		auto o = op_type{};
		auto v = size_type{};
		constexpr auto zero = size_type{0};
		for(k[r] = 0ul; k[r] < e.at(r); ++k[r])
			for(k[o(r,1)] = 0ul; k[o(r,1)] < e.at(o(r,1)); ++k[o(r,1)])
				for(k[o(r,2)] = 0ul; k[o(r,2)] < e.at(o(r,2)); ++k[o(r,2)], ++v)
					BOOST_CHECK_EQUAL(ublas::detail::access<zero>(zero, w, k[0], k[1],k[2]), v);

	};

	auto check4 = [](const shape_type& e, const strides_type& w)
	{
		std::array<unsigned,4> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::tag::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::tag::first_order> ? 3 : 0;
		auto o = op_type{};
		auto v = size_type{};
		constexpr auto zero = size_type{0};
		for(k[r] = 0ul; k[r] < e.at(r); ++k[r])
			for(k[o(r,1)] = 0ul; k[o(r,1)] < e.at(o(r,1)); ++k[o(r,1)])
				for(k[o(r,2)] = 0ul; k[o(r,2)] < e.at(o(r,2)); ++k[o(r,2)])
					for(k[o(r,3)] = 0ul; k[o(r,3)] < e.at(o(r,3)); ++k[o(r,3)], ++v)
						BOOST_CHECK_EQUAL(ublas::detail::access<zero>(zero, w, k[0], k[1],k[2], k[3]), v);
	};

	auto check = [check1,check2,check3,check4](auto const& e) {
		auto w = strides_type(e);

				 if(e.size() == 1) check1(e,w);
		else if(e.size() == 2) check2(e,w);
		else if(e.size() == 3) check3(e,w);
		else if(e.size() == 4) check4(e,w);

	};

	for(auto const& e : extents)
		check(e);
}


BOOST_AUTO_TEST_SUITE_END();

