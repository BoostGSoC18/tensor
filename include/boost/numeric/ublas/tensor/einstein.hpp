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

namespace boost {
namespace numeric {
namespace ublas {
namespace placeholders {

// Adapter
template<std::size_t I>
struct Placeholder { static constexpr auto value = I; };

extern const Placeholder< 1> a;
extern const Placeholder< 2> b;
extern const Placeholder< 3> c;
extern const Placeholder< 4> d;
extern const Placeholder< 5> e;


} // namespace placeholders


//template<std::size_t N>
//struct EinsteinIndices
//{

//	EinsteinIndices();

//	std::array<std::size_t, N> _indices;
//};




//template<std::size_t N, class T>
//class TensorMultiplier
//{
//public:
//	static constexpr size_type n = N;
//	using size_type  = std::size_t;
//	using array_type = std::array<size_type,n>;

//private:
//	TensorMultiplier() = delete;
//	TensorMultiplier(array_type const& indices) : _indices(indices)
//	{

//	}


//	array_type _indices;

//};



} // namespace ublas
} // namespace numeric
} // namespace boost

#endif
