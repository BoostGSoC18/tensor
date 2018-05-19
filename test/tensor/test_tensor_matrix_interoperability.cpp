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



#include <random>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor_matrix_interoperability, * boost::unit_test::depends_on("test_tensor") ) ;


using test_types = zip<int,long,float,double>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	tensor_type a1 = matrix_type();
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	tensor_type a2 = matrix_type(1,1);
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	tensor_type a3 = matrix_type(2,1);
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	tensor_type a4 = matrix_type(1,2);
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	tensor_type a5 = matrix_type(2,3);
	BOOST_CHECK_EQUAL(  a5.size() , 6 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using vector_type = typename tensor_type::vector_type;

	tensor_type a1 = vector_type();
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	tensor_type a2 = vector_type(1);
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	tensor_type a3 = vector_type(2);
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	tensor_type a4 = vector_type(2);
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	tensor_type a5 = vector_type(3);
	BOOST_CHECK_EQUAL(  a5.size() , 3 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);
}


struct fixture_matrix {
	using extents_type = boost::numeric::ublas::basic_extents<std::size_t>;
	fixture_matrix() : extents{
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{2,3}, // 4
				extents_type{9,7}, // 5
				}
	{}
	std::vector<extents_type> extents;
};

struct fixture_vector {
	using extents_type = boost::numeric::ublas::basic_extents<std::size_t>;
	fixture_vector() : extents{
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{1,3}, // 4
				extents_type{4,1}, // 5
				}
	{}
	std::vector<extents_type> extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor_extents, value,  test_types, fixture_matrix )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const& e) {
		assert(e.size()==2);
		tensor_type t = matrix_type{e[0],e[1]};
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );
		BOOST_CHECK_NE    (  t.data() , nullptr);
	};

	for(auto const& e : extents)
		check(e);
}


//BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor_extents, value,  test_types, fixture_vector )
//{
//	using namespace boost::numeric;
//	using value_type  = typename value::first_type;
//	using layout_type = typename value::second_type;
//	using tensor_type = ublas::tensor<value_type, layout_type>;
//	using vector_type = typename tensor_type::vector_type;

//	auto check = [](auto const& e) {
//		assert(e.size()==2);
//		if(e.empty())
//			return;
//		tensor_type t = vector_type(e.product());
//		BOOST_CHECK_EQUAL (  t.size() , e.product() );
//		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
//		BOOST_CHECK       ( !t.empty()    );
//		BOOST_CHECK_NE    (  t.data() , nullptr);
//	};

//	for(auto const& e : extents)
//		check(e);
//}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_interoperability_copy_assignment, value,  test_types, fixture_matrix )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const& e)
	{
		assert(e.size() == 2);
		auto t = tensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		t = r;

		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );
		BOOST_CHECK_NE    (  t.data() , nullptr);

		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), r(i,j)  );
			}
		}
	};

	for(auto const& e : extents)
		check(e);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_interoperability_move_assignment, value,  test_types, fixture_matrix )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const& e)
	{
		assert(e.size() == 2);
		auto t = tensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		auto q = r;
		t = std::move(r);

		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );
		BOOST_CHECK_NE    (  t.data() , nullptr);

		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), q(i,j)  );
			}
		}
	};

	for(auto const& e : extents)
		check(e);
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_interoperability_matrix_expressions, value,  test_types, fixture_matrix )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const& e)
	{
		assert(e.size() == 2);
		auto t = tensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		t = r + 3*r;
		tensor_type s = r + 3*r;
		tensor_type q = s + tensor_type(r + 3*r) + s; // + 3*r


		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );
		BOOST_CHECK_NE    (  t.data() , nullptr);

		BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  s.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  s.size() , e.product() );
		BOOST_CHECK_EQUAL (  s.rank() , e.size() );
		BOOST_CHECK       ( !s.empty()    );
		BOOST_CHECK_NE    (  s.data() , nullptr);

		BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  q.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  q.size() , e.product() );
		BOOST_CHECK_EQUAL (  q.rank() , e.size() );
		BOOST_CHECK       ( !q.empty()    );
		BOOST_CHECK_NE    (  q.data() , nullptr);


		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), 4*r(i,j)  );
				BOOST_CHECK_EQUAL( s.at(i,j), t.at(i,j)  );
				BOOST_CHECK_EQUAL( q.at(i,j), 3*s.at(i,j)  );
			}
		}
	};

	for(auto const& e : extents)
		check(e);
}


BOOST_AUTO_TEST_SUITE_END();

