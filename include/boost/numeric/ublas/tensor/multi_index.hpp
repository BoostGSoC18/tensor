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

#ifndef _BOOST_UBLAS_TENSOR_MULTI_INDEX_HPP_
#define _BOOST_UBLAS_TENSOR_MULTI_INDEX_HPP_


#include <cstddef>
#include <array>
#include <vector>


namespace boost {
namespace numeric {
namespace ublas {

template<class V, class S, class A>
class tensor;

}
}
}

namespace boost {
namespace numeric {
namespace ublas {
namespace indices {

// Adapter

template<std::size_t I>
struct index { static constexpr std::size_t value = I; };

static constexpr index< 0> _;
static constexpr index< 1> _a;
static constexpr index< 2> _b;
static constexpr index< 3> _c;
static constexpr index< 4> _d;
static constexpr index< 5> _e;
static constexpr index< 6> _f;
static constexpr index< 7> _g;
static constexpr index< 8> _h;
static constexpr index< 9> _i;
static constexpr index<10> _j;
static constexpr index<11> _k;
static constexpr index<12> _l;
static constexpr index<13> _m;
static constexpr index<14> _n;
static constexpr index<15> _o;
static constexpr index<16> _p;
static constexpr index<17> _q;
static constexpr index<18> _r;
static constexpr index<19> _s;
static constexpr index<20> _t;
static constexpr index<21> _u;
static constexpr index<22> _v;
static constexpr index<23> _w;
static constexpr index<24> _x;
static constexpr index<25> _y;
static constexpr index<26> _z;


} // namespace indices
}
}
}


namespace boost {
namespace numeric {
namespace ublas {

template<std::size_t N>
class multi_index
{

	using size_type  = std::size_t;
	template<std::size_t I>
	using index_type = indices::index<I>;
	using array_type = std::array<std::size_t, N>;

public:
	multi_index() = delete;

	template<std::size_t I, class ... indexes>
	constexpr
	multi_index(index_type<I> const& i, indexes ... is )
			 : _indices{getindex(i), getindex(is)... }
	{
		static_assert( sizeof...(is)+1 == N,
					   "Static assert in boost::numeric::ublas::multi_index: number of constructor arguments is not equal to the template parameter." );
		if( ! valid(i,is...) )
			throw std::runtime_error("Error in boost::numeric::ublas::multi_index: constructor arguments are not valid." );
	}

	multi_index(multi_index const& other)
		: _indices(other._indices)
	{
	}

	multi_index& operator=(multi_index const& other)
	{
		this->_indices = other._indices;
		return *this;
	}

	~multi_index() = default;

	auto const& indices() const { return _indices; }
	constexpr auto size() const { return _indices.size(); }

	auto at(std::size_t i) const { return _indices.at(i); }

private:
	template<std::size_t I>
	constexpr auto getindex(index_type<I> const& i) { return i.value; }


	template<std::size_t I, std::size_t J, class ... indexes>
	static constexpr bool has_i (index_type<I> i, index_type<J> j, indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		constexpr auto b = (i.value==j.value && i.value != 0);

		if constexpr (n>0)
			return b && has_i( i, is ... );
		else
			return b;
	}

	template<std::size_t I, class ... indexes>
	static constexpr bool valid (index_type<I> i, indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		if constexpr (n>0)
			return !has_i( i, is ... ) && valid(is...);
		else
			return true;
	}

	array_type _indices;
};

}
}
}


//namespace boost {
//namespace numeric {
//namespace ublas {
//namespace detail {

//template<class V, class S, class A, std::size_t N>
//using tensor_multiindex_pair = std::pair< tensor<V,S,A> const&, multi_index<N> >;

//template<std::size_t N, std::size_t M>
//auto extract_corresponding_indices(
//		multi_index<N> const& lhs_multi_index,
//		multi_index<M> const& rhs_multi_index)
//{
//	using vtype = std::vector<std::size_t>;

//	auto pp = std::make_pair( vtype {}, vtype{}  );

//	for(auto i = 0u; i < N; ++i)
//		for(auto j = 0u; j < M; ++j)
//			if ( lhs_multi_index.at(i) == rhs_multi_index.at(j) && lhs_multi_index.at(i) != indices::_.value)
//				pp.first .push_back( i+1 ),
//				pp.second.push_back( j+1 );

//	if(pp.first.empty())
//		throw std::runtime_error("Error in boost::numeric::ublas::extract_corresponding_indices: number of contracting indices of lhs_multi_index is zero.");

//	if(pp.first.size() != pp.second.size())
//		throw std::runtime_error("Error in boost::numeric::ublas::extract_corresponding_indices: number of contracting indices from lhs_multi_index and rhs_multi_index must be equal.");

//	return pp;
//}


//} // namespace detail
//} // namespace ublas
//} // namespace numeric
//} // namespace boost


#endif // MULTI_INDEX_HPP
