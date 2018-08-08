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
namespace index {

/** @brief Proxy template class for the einstein summation notation
 *
 * @note index::index_type<K> for 0<=K<=16 is used in tensor::operator()
 *
 * @tparam I wrapped integer
*/
template<std::size_t I>
struct index_type
{
	static constexpr std::size_t value = I;

	constexpr bool operator == (std::size_t other) const { return value == other; }
	constexpr bool operator != (std::size_t other) const { return value != other; }

	template <std::size_t K>
	constexpr bool operator == (index_type<K> /*other*/) const {  return I==K; }
	template <std::size_t  K>
	constexpr bool operator != (index_type<K> /*other*/) const {  return I!=K; }

	std::size_t operator()() const { return I; }
};

/** @brief Proxy classes for the einstein summation notation
 *
 * @note index::_a ... index::_z is used in tensor::operator()
*/

static constexpr index_type< 0> _;
static constexpr index_type< 1> _a;
static constexpr index_type< 2> _b;
static constexpr index_type< 3> _c;
static constexpr index_type< 4> _d;
static constexpr index_type< 5> _e;
static constexpr index_type< 6> _f;
static constexpr index_type< 7> _g;
static constexpr index_type< 8> _h;
static constexpr index_type< 9> _i;
static constexpr index_type<10> _j;
static constexpr index_type<11> _k;
static constexpr index_type<12> _l;
static constexpr index_type<13> _m;
static constexpr index_type<14> _n;
static constexpr index_type<15> _o;
static constexpr index_type<16> _p;
static constexpr index_type<17> _q;
static constexpr index_type<18> _r;
static constexpr index_type<19> _s;
static constexpr index_type<20> _t;
static constexpr index_type<21> _u;
static constexpr index_type<22> _v;
static constexpr index_type<23> _w;
static constexpr index_type<24> _x;
static constexpr index_type<25> _y;
static constexpr index_type<26> _z;


} // namespace indices
}
}
}


namespace boost {
namespace numeric {
namespace ublas {


/** @brief Proxy class for the einstein summation notation
 *
 * Denotes an array of index_type types ::_a for 0<=K<=16 is used in tensor::operator()
*/
template<std::size_t N>
class multi_index
{
public:
	multi_index() = delete;

	template<std::size_t I, class ... indexes>
	constexpr multi_index(index::index_type<I> const& i, indexes ... is )
			 : _base{i(), getindex(is)... }
	{
		static_assert( sizeof...(is)+1 == N,
					   "Static assert in boost::numeric::ublas::multi_index: number of constructor arguments is not equal to the template parameter." );
		if( ! valid(i,is...) )
			throw std::runtime_error("Error in boost::numeric::ublas::multi_index: constructor arguments are not valid." );
	}

	multi_index(multi_index const& other)
		: _base(other._base)
	{
	}

	multi_index& operator=(multi_index const& other)
	{
		this->_base = other._base;
		return *this;
	}

	~multi_index() = default;

	auto const& base() const { return _base; }
	constexpr auto size() const { return _base.size(); }

	constexpr auto at(std::size_t i) const { return _base.at(i); }

	constexpr auto operator[](std::size_t i) const { return _base.at(i); }

private:
	template<std::size_t I>
	constexpr auto getindex(index::index_type<I> ) const { return I; }


	template<std::size_t I, std::size_t J, class ... indexes>
	static constexpr bool has_i (index::index_type<I> i, index::index_type<J> j, indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		constexpr auto b = (i==j && i!=0ul);

		if constexpr (n>0)
			return b && has_i( i, is ... );
		else
			return b;
	}

	template<std::size_t I, class ... indexes>
	static constexpr bool valid (index::index_type<I> i, indexes ... is )
	{
		constexpr auto n = sizeof...(is);
		if constexpr (n>0)
			return !has_i( i, is ... ) && valid(is...);
		else
			return true;
	}

	std::array<std::size_t, N> _base;
};

template<std::size_t K, std::size_t N>
static constexpr auto get(multi_index<N> const& m) { return std::get<K>(m.base()); }

}
}
}
#endif // MULTI_INDEX_HPP
