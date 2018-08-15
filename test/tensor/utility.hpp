//  Copyright (c) 2018
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

#ifndef _BOOST_UBLAS_TEST_TENSOR_UTILITY_
#define _BOOST_UBLAS_TEST_TENSOR_UTILITY_



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
// creates e.g.
// using test_types = zip<long,float>::with_t<first_order,last_order>; // equals
// using test_types = std::tuple< std::pair<float, first_order>, std::pair<float, last_order >, std::pair<double,first_order>, std::pair<double,last_order >>;
// static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
// static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::tag::first_order>::value,"should be boost::numeric::ublas::tag::first_order ");




template<class ... zipped_types>
struct tensor_types_helper;

template<template<class,class> class tensor_type, class pair_type>
struct tensor_types_helper<
		tensor_type<typename pair_type::first_type, typename pair_type::second_type>,
		std::tuple<pair_type>>
{
	using type = tensor_type<typename pair_type::first_type, typename pair_type::second_type>;
};

template<template<class,class> class tensor_type, class pair_type, class ... pair_types>
struct tensor_types_helper<
		tensor_type<typename pair_type::first_type, typename pair_type::second_type>,
		std::tuple<pair_type, pair_types...>>
{
	using ttype = tensor_type<typename pair_type::first_type, typename pair_type::second_type>;
	using next_type = typename tensor_types_helper<ttype, std::tuple<pair_types...>>::type;
	using type = std::tuple<  tensor_types_helper<ttype, pair_type>, next_type  >;
};


// expecting sth like std::tuple< std::pair<float, first_order>, std::pair<float, last_order >, std::pair<double,first_order>, std::pair<double,last_order >>;
template<template<class,class> class tensor_type, class tuple_type>
struct tensor_types
{
	using tpair = std::tuple_element_t<0,tuple_type>;
	using ttype = tensor_type<std::tuple_element_t<0,tpair>,std::tuple_element_t<1,tpair>>;
	using type = typename tensor_types_helper<ttype, tuple_type>::type;
};



#endif
