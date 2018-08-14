//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen Germany
//


#ifndef _BOOST_UBLAS_TENSOR_ALGORITHMS_TRAITS_HPP_
#define _BOOST_UBLAS_TENSOR_ALGORITHMS_TRAITS_HPP_


#include <stdexcept>
#include <complex>
#include <functional>
#include <tuple>

#include "tags.hpp"

namespace boost {
namespace numeric {
namespace ublas {

// unit_tag


namespace detail {


template<class unsigned_type, class layout_tag, class access_tag>
struct increment_pointer_type__;

template<class unsigned_type>
struct increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::first_order,tag::unit_access>
{
	unsigned_type p;
	template<class value_type, class size_type>
	inline void operator()(value_type*      a, size_type const*const) { ++a;	}

//	template<class value_type, class size_type>
//	inline void operator()(value_type*const a, size_type const*const) { ++a;	}
};


template<class unsigned_type>
struct increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::first_order,tag::non_unit_access>
{
	unsigned_type p;
	template<class value_type, class size_type>
	inline void operator()(value_type*      a, size_type const*const w) { a+=w[0]; }
//	template<class value_type, class size_type>
//	inline void operator()(value_type*const a, size_type const*const w) { a+=w[0]; }
};

template<class unsigned_type>
struct increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::last_order,tag::unit_access>
{
	unsigned_type p;
	template<class value_type, class size_type>
	inline void operator()(value_type*      a, size_type const*const) { ++a;	}
//	template<class value_type, class size_type>
//	inline void operator()(value_type*const a, size_type const*const) { ++a;	}
};


template<class unsigned_type>
struct increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::last_order,tag::non_unit_access>
{
	unsigned_type p;
	template<class value_type, class size_type>
	inline void operator()(value_type*      a, size_type const*const w) { a+=w[p]; }
//	template<class value_type, class size_type>
//	inline void operator()(value_type*const a, size_type const*const w) { a+=w[p]; }
};




///////////////////////////////////
///////////////////////////////////
///////////////////////////////////



template<class unsigned_type, class layout_tag, class access_tag>
struct recursive_function_traits;





template<class unsigned_type, class access_tag>
struct recursive_function_traits<unsigned_type, boost::numeric::ublas::tag::first_order, access_tag>
{
	struct compare_type
	{ unsigned_type p; inline auto operator()(unsigned_type r) { return r>0; } };

	struct increment_recursion_type
	{ inline auto operator()(unsigned_type r) { return r-1;} };

	using increment_pointer_type = detail::increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::first_order,access_tag>;

};

template<class unsigned_type, class access_tag>
struct recursive_function_traits<unsigned_type, boost::numeric::ublas::tag::last_order, access_tag>
{
	struct compare_type
	{
		unsigned_type p;
		inline auto operator()(unsigned_type r) { return r<p; }
	};

	struct increment_recursion_type
	{
		inline auto operator()(unsigned_type r) { return r+1;}
	};

	using increment_pointer_type = detail::increment_pointer_type__<unsigned_type,boost::numeric::ublas::tag::last_order,access_tag>;

};

} // detail
}
}
}

#endif
