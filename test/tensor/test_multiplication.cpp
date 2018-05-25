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
#include <vector>
#include <boost/numeric/ublas/tensor/multiplication.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE (test_tensor_contraction) ;


using test_types = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;


struct fixture {
	using extents_type = boost::numeric::ublas::shape;
	fixture() : extents {
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{2,3}, // 4
				extents_type{5,4}, // 5
				extents_type{2,3,1}, // 6
				extents_type{4,1,3}, // 7
				extents_type{1,2,3}, // 8
				extents_type{4,2,3}, // 9
				extents_type{4,2,3,5} // 10
				}
	{}
	std::vector<extents_type> extents;
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_mtv, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::shape;
	using extents_type_base = typename extents_type::base_type;


	for(auto const& na : extents) {

		if(na.size() > 2)
			continue;

		auto a = vector_type(na.product(), value_type{2});
		auto wa = strides_type(na);
		for(auto m = 0ul; m < na.size(); ++m){
			auto nb = extents_type {na[m],1};
			auto wb = strides_type (nb);
			auto b  = vector_type  (nb.product(), value_type{1} );

			auto nc_base = extents_type_base(std::max(na.size()-1,2ul),1);

			for(auto i = 0ul, j = 0ul; i < na.size(); ++i)
				if(i != m)
					nc_base[j++] = na[i];

			auto nc = extents_type (nc_base);
			auto wc = strides_type (nc);
			auto c  = vector_type  (nc.product(), value_type{0});

			ublas::detail::recursive::mtv(
						m,
						c.data(), nc.data(), wc.data(),
						a.data(), na.data(), wa.data(),
						b.data());


			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(na[m]) * a[i] );

		}
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_mtm, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::shape;
	using extents_type_base = typename extents_type::base_type;


	for(auto const& na : extents) {

		if(na.size() != 2)
			continue;

		auto a  = vector_type  (na.product(), value_type{2});
		auto wa = strides_type (na);

		auto nb = extents_type {na[1],na[0]};
		auto wb = strides_type (nb);
		auto b  = vector_type  (nb.product(), value_type{1} );

		auto nc = extents_type {na[0],nb[1]};
		auto wc = strides_type (nc);
		auto c  = vector_type  (nc.product());


		ublas::detail::recursive::mtm(
					c.data(), nc.data(), wc.data(),
					a.data(), na.data(), wa.data(),
					b.data(), nb.data(), wb.data());


		for(auto i = 0u; i < c.size(); ++i)
			BOOST_CHECK_EQUAL( c[i] , value_type(na[1]) * a[0] );


	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttv, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::shape;
	using extents_type_base = typename extents_type::base_type;


	for(auto const& na : extents) {

		auto a = vector_type(na.product(), value_type{2});
		auto wa = strides_type(na);
		for(auto m = 0ul; m < na.size(); ++m){
			auto b  = vector_type  ( na[m], value_type{1} );
			auto nb = extents_type {na[m],1};
			auto wb = strides_type (nb);

			auto nc_base = extents_type_base(std::max(na.size()-1,2ul),1);

			for(auto i = 0ul, j = 0ul; i < na.size(); ++i)
				if(i != m)
					nc_base[j++] = na[i];

			auto nc = extents_type (nc_base);
			auto wc = strides_type (nc);
			auto c  = vector_type  (nc.product(), value_type{0});

			ublas::ttv( m+1, na.size(),
									c.data(), nc.data(), wc.data(),
									a.data(), na.data(), wa.data(),
									b.data(), nb.data(), wb.data());


			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(na[m]) * a[i] );

		}
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttm, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::shape;


	for(auto const& na : extents) {

		auto a = vector_type(na.product(), value_type{2});
		auto wa = strides_type(na);
		for(auto m = 0ul; m < na.size(); ++m){
			auto nb = extents_type { na[m], na[m] };
			auto b  = vector_type  ( nb.product(), value_type{1} );
			auto wb = strides_type (nb);


			auto nc = na;
			auto wc = strides_type (nc);
			auto c  = vector_type  (nc.product(), value_type{0});

			ublas::ttm( m+1, na.size(),
									c.data(), nc.data(), wc.data(),
									a.data(), na.data(), wa.data(),
									b.data(), nb.data(), wb.data());

			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(na[m]) * a[i] );

		}
	}
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;


	for(auto const& n : extents) {

		auto a = vector_type(n.product(), value_type{2});
		auto b = vector_type(n.product(), value_type{3});
		auto w = strides_type(n);

		auto c = ublas::inner(n.size(), n.data(), a.data(), w.data(), b.data(), w.data(), value_type(0));
		auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_type(0));


		BOOST_CHECK_EQUAL( c , cref );

	}

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using extents_type = ublas::shape;
	using strides_type = ublas::strides<layout_type>;
	using vector_type  = std::vector<value_type>;


	for(auto const& na : extents) {

		auto a = vector_type(na.product(), value_type{2});
		auto wa = strides_type(na);

		for(auto const& nb : extents) {

			auto b = vector_type(nb.product(), value_type{3});
			auto wb = strides_type(nb);

			auto c = vector_type(nb.product()*na.product());
			auto nc = typename extents_type::base_type(na.size()+nb.size());

			for(auto i = 0u; i < na.size(); ++i)
				nc[i] = na[i];
			for(auto i = 0u; i < nb.size(); ++i)
				nc[i+na.size()] = nb[i];

			auto wc = strides_type(extents_type(nc));

			ublas::outer(c.data(), nc.size(), nc.data(), wc.data(),
									 a.data(), na.size(), na.data(), wa.data(),
									 b.data(), nb.size(), nb.data(), wb.data());

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
		}

	}

}


BOOST_AUTO_TEST_SUITE_END();

