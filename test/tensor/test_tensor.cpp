//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestTensor
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE ( test_tensor, * boost::unit_test::depends_on("test_extents") ) ;


using test_types =
std::tuple<
std::pair<float, boost::numeric::ublas::first_order>,
std::pair<float, boost::numeric::ublas::last_order >,
std::pair<double,boost::numeric::ublas::first_order>,
std::pair<double,boost::numeric::ublas::last_order >
>;


template<class ... types>
struct zip_helper;

template<class type1, class ... types3>
struct zip_helper<std::tuple<types3...>, type1>
{
	template<class ... types2>
	struct with
	{
		using type = std::tuple<types3...,std::pair<type1,types2>...>;
	};
	template<class ... types2>
	using with_t = typename with<types2...>::type;
};


template<class type1, class ... types3, class ... types1>
struct zip_helper<std::tuple<types3...>, type1, types1...>
{
	template<class ... types2>
	struct with
	{
		using next_tuple = std::tuple<types3...,std::pair<type1,types2>...>;
		using type       = typename zip_helper<next_tuple, types1...>::template with<types2...>::type;
	};

	template<class ... types2>
	using with_t = typename with<types2...>::type;
};

template<class ... types>
using zip = zip_helper<std::tuple<>,types...>;

using test_types2 = zip<float,double>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;



static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::first_order>::value,"should be boost::numeric::ublas::first_order ");
static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<1,test_types2>>, float>::value,"should be float ");
static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<1,test_types2>>, boost::numeric::ublas::last_order>::value,"should be boost::numeric::ublas::last_order ");
static_assert(std::is_same<test_types2, test_types>::value,"not same");


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;


	auto a1 = tensor_type{};
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	auto a2 = tensor_type{1,1};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = tensor_type{2,1};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = tensor_type{1,2};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = tensor_type{2,1};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = tensor_type{4,3,2};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = tensor_type{4,1,2};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);


}



BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto a2 = tensor_type{ublas::extents{1,1}};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = tensor_type{ublas::extents{2,1}};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = tensor_type{ublas::extents{1,2}};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = tensor_type{ublas::extents{2,1}};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = tensor_type{ublas::extents{4,3,2}};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = tensor_type{ublas::extents{4,1,2}};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_layout, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto a2 = tensor_type{ublas::extents{1,1}};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = tensor_type{ublas::extents{2,1}};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = tensor_type{ublas::extents{1,2}};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = tensor_type{ublas::extents{2,1}};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = tensor_type{ublas::extents{4,3,2}};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = tensor_type{ublas::extents{4,1,2}};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto a2 = tensor_type{ublas::extents{1,1}, ublas::unbounded_array<typename value::first_type>(1,1)  };
	BOOST_CHECK_EQUAL( a2[0], 1 );

//	BOOST_CHECK_THROW( ublas::tensor<value>{ublas::extents{1,1}, ublas::unbounded_array<value>(1)  }; )



//	BOOST_CHECK_EQUAL(  a2.size() , 1 );
//	BOOST_CHECK( !a2.empty() );
//	BOOST_CHECK_NE(  a2.data() , nullptr);

//	auto a3 = ublas::tensor<value>{ublas::extents{2}, ublas::layout{1}};
//	BOOST_CHECK_EQUAL(  a3.size() , 2 );
//	BOOST_CHECK( !a3.empty() );
//	BOOST_CHECK_NE(  a3.data() , nullptr);

//	auto a4 = ublas::tensor<value>{ublas::extents{1,2}, ublas::layout{1,2}};
//	BOOST_CHECK_EQUAL(  a4.size() , 2 );
//	BOOST_CHECK( !a4.empty() );
//	BOOST_CHECK_NE(  a4.data() , nullptr);

//	auto a5 = ublas::tensor<value>{ublas::extents{2,1}, ublas::layout{2,1}};
//	BOOST_CHECK_EQUAL(  a5.size() , 2 );
//	BOOST_CHECK( !a5.empty() );
//	BOOST_CHECK_NE(  a5.data() , nullptr);

//	auto a6 = ublas::tensor<value>{ublas::extents{4,3,2}, ublas::layout{3,2,1}};
//	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
//	BOOST_CHECK( !a6.empty() );
//	BOOST_CHECK_NE(  a6.data() , nullptr);

//	auto a7 = ublas::tensor<value>{ublas::extents{4,1,2}, ublas::layout{1,2,3}};
//	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
//	BOOST_CHECK( !a7.empty() );
//	BOOST_CHECK_NE(  a7.data() , nullptr);
}



BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_access, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;


	auto a2 =tensor_type{3,1};
	for(auto i = 0u; i < a2.size(); ++i){
		a2[i] = static_cast<value_type>(i);
		BOOST_CHECK_EQUAL( a2[i], static_cast<value_type>(i) );
	}

	auto a3 = tensor_type{3,3};
	for(auto i = 0u; i < a3.size(); ++i){
		a3[i] = static_cast<value_type>(i);
		BOOST_CHECK_EQUAL( a3[i], static_cast<value_type>(i) );
	}

}

BOOST_AUTO_TEST_SUITE_END();
