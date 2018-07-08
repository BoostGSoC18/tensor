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
namespace placeholders {

// Adapter

template<std::size_t I>
struct Placeholder
{ static constexpr std::size_t value = I; };

static constexpr Placeholder< 0> _;
static constexpr Placeholder< 1> a;
static constexpr Placeholder< 2> b;
static constexpr Placeholder< 3> c;
static constexpr Placeholder< 4> d;
static constexpr Placeholder< 5> e;
static constexpr Placeholder< 6> f;
static constexpr Placeholder< 7> g;
static constexpr Placeholder< 8> h;
static constexpr Placeholder< 9> i;
static constexpr Placeholder<10> j;
static constexpr Placeholder<11> k;
static constexpr Placeholder<12> l;


} // namespace placeholders


template<std::size_t N>
struct MultiplicationIndices
{
	using size_type  = std::size_t;
	template<std::size_t I>
	using pindex_type = placeholders::Placeholder<I>;
	using array_type = std::array<std::size_t, N>;

	MultiplicationIndices() = delete;

	template<std::size_t I, class ... Placeholders>
	MultiplicationIndices(pindex_type<I> const& p, Placeholders ... ps )
			 : _indices{getIndex(p), getIndex(ps)... }
	{
		static_assert( sizeof...(ps)+1 == N, " " );
	}


	MultiplicationIndices(MultiplicationIndices const& other)
		: _indices(other._indices)
	{
	}

	MultiplicationIndices& operator=(MultiplicationIndices const& other)
	{
		_indices = (other._indices);
	}

	~MultiplicationIndices() = default;

	template<std::size_t I>
	constexpr auto getIndex(pindex_type<I> const& p)
	{
		return p.value;
	}

	auto const& indices() const { return _indices; }

	auto at(std::size_t i) const { return _indices.at(i); }

	array_type _indices;
	// array of index identifier, e.g. c<3>,a<1>,b<2>,...
};

template<std::size_t N, std::size_t M>
auto corresponding(MultiplicationIndices<N> const& lhs,
				   MultiplicationIndices<M> const& rhs)
{
	using vtype = std::vector<std::size_t>;

	auto pp = std::make_pair( vtype {}, vtype{}  );

	for(auto i = 0u; i < N; ++i)
		for(auto j = 0u; j < M; ++j)
			if ( lhs.at(i) == rhs.at(j) )
				pp.first .push_back( i+1 ),
				pp.second.push_back( j+1 );


	assert(pp.first.size() == pp.second.size());

	assert(!pp.first.empty());

	return pp;

}




} // namespace ublas
} // namespace numeric
} // namespace boost

#endif
