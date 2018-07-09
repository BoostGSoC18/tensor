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


#ifndef _BOOST_UBLAS_TENSOR_EINSTEIN_NOTATION_
#define _BOOST_UBLAS_TENSOR_EINSTEIN_NOTATION_

#include <cstddef>
#include <array>
#include <vector>


namespace boost {
namespace numeric {
namespace ublas {
namespace indices {

// Adapter

template<std::size_t I>
struct Index
{ static constexpr std::size_t value = I; };

static constexpr Index< 0> _;
static constexpr Index< 1> _a;
static constexpr Index< 2> _b;
static constexpr Index< 3> _c;
static constexpr Index< 4> _d;
static constexpr Index< 5> _e;
static constexpr Index< 6> _f;
static constexpr Index< 7> _g;
static constexpr Index< 8> _h;
static constexpr Index< 9> _i;
static constexpr Index<10> _j;
static constexpr Index<11> _k;
static constexpr Index<12> _l;
static constexpr Index<13> _m;
static constexpr Index<14> _n;
static constexpr Index<15> _o;
static constexpr Index<16> _p;
static constexpr Index<17> _q;
static constexpr Index<18> _r;
static constexpr Index<19> _s;
static constexpr Index<20> _t;
static constexpr Index<21> _u;
static constexpr Index<22> _v;
static constexpr Index<23> _w;
static constexpr Index<24> _x;
static constexpr Index<25> _y;
static constexpr Index<26> _z;


} // namespace indices


template<std::size_t N>
class MIndices
{

	using size_type  = std::size_t;
	template<std::size_t I>
	using index_type = indices::Index<I>;
	using array_type = std::array<std::size_t, N>;

public:
	MIndices() = delete;

	template<std::size_t I, class ... Indexes>
	constexpr
	MIndices(index_type<I> const& i, Indexes ... is )
			 : _indices{getIndex(i), getIndex(is)... }
	{
		static_assert( sizeof...(is)+1 == N, "Static assert in boost::numeric::ublas::MIndices: number of constructor arguments is not equal to the template parameter." );
		if( ! valid(i,is...) )
			throw std::runtime_error("Error in boost::numeric::ublas::MIndices: constructor arguments are not valid." );
	}

	MIndices(MIndices const& other)
		: _indices(other._indices)
	{
	}

	MIndices& operator=(MIndices const& other)
	{
		this->_indices = other._indices;
		return *this;
	}

	bool operator==(MIndices const& other) const
	{
		return this->_indices == other._indices;
	}

	bool operator!=(MIndices const& other) const
	{
		return this->_indices != other._indices;
	}

	~MIndices() = default;

	auto const& indices() const { return _indices; }

	auto at(std::size_t i) const { return _indices.at(i); }

private:
	template<std::size_t I>
	constexpr auto getIndex(index_type<I> const& i) { return i.value; }


	template<std::size_t I, std::size_t J, class ... Indexes>
	static constexpr bool has_i (index_type<I> i, index_type<J> j, Indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		constexpr auto b = (i.value==j.value);

		if constexpr (n>0)
			return b && has_i( i, is ... );
		else
			return b;
	}

	template<std::size_t I, class ... Indexes>
	static constexpr bool valid (index_type<I> i, Indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		if constexpr (n>0)
			return !has_i( i, is ... ) && valid(is...);
		else
			return true;
	}

	array_type _indices;
};

template<std::size_t N, std::size_t M>
auto extract_corresponding_indices(
		MIndices<N> const& lhs,
		MIndices<M> const& rhs)
{
	using vtype = std::vector<std::size_t>;

	auto pp = std::make_pair( vtype {}, vtype{}  );

	for(auto i = 0u; i < N; ++i)
		for(auto j = 0u; j < M; ++j)
			if ( lhs.at(i) == rhs.at(j) && lhs.at(i) != indices::_.value)
				pp.first .push_back( i+1 ),
				pp.second.push_back( j+1 );

	if(pp.first.empty())
		throw std::runtime_error("Error in boost::numeric::ublas::extract_corresponding_indices: number of contracting indices of lhs indices is zero.");

	if(pp.first.size() != pp.second.size())
		throw std::runtime_error("Error in boost::numeric::ublas::extract_corresponding_indices: number of contracting indices from lhs and rhs indices must be equal.");


	return pp;

}





} // namespace ublas
} // namespace numeric
} // namespace boost

#endif
