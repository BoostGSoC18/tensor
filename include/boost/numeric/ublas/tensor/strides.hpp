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


#ifndef _BOOST_UBLAS_STRIDES_
#define _BOOST_UBLAS_STRIDES_

#include <vector>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>

#include <cassert>

namespace boost { namespace numeric { namespace ublas {

template<class T, class A>
class basic_extents;

template<class T, class A>
class basic_layout;

template<class __int_type, class __allocator_type = std::allocator<__int_type>>
class basic_strides
{
	using base_type = std::vector<__int_type, __allocator_type>;
	static_assert( std::numeric_limits<typename base_type::value_type>::is_integer, "Static error in basic_layout: type must be of type integer.");
	static_assert(!std::numeric_limits<typename base_type::value_type>::is_signed,  "Static error in basic_layout: type must be of type unsigned integer.");

public:
	using value_type = typename base_type::value_type;
	using reference = typename base_type::reference;
	using const_reference = typename base_type::const_reference;
	using size_type = typename base_type::size_type;
	//using pointer = typename base_type::pointer;
	using const_pointer = typename base_type::const_pointer;
	using const_iterator = typename base_type::const_iterator;


	constexpr explicit basic_strides()
		: _base{}
	{
	}

	template <class T1, class T2, class A1, class A2>
	basic_strides(basic_extents<T1,A1> const& s, basic_layout<T2,A2> const& l)
	    : _base(s.size(),1)
	{
		if(s.size() != l.size())
			throw std::runtime_error("Error in basic_strides::basic_strides() : shape.size() != layout.size()");

		if(!s.valid() || !l.valid())
			throw std::runtime_error("Error in basic_strides::basic_strides() : shape or layout not valid");


		if(s.is_vector() || s.is_scalar()){
			_base[0] = 1;
			_base[1] = 1;
			return;
		}

		const size_t end = s.size();

		_base[l[0]-1] = 1u;

		for(size_t r = 1; r < end; ++r)
		{
			const size_t pr   = l[r]-1;
			const size_t pr_1 = l[r-1]-1;
			_base[pr] = _base[pr_1] * s[pr_1];
		}
	}

	basic_strides(basic_strides const& l )
	    : _base(l._base)
	{}

	basic_strides(basic_strides && l )
	    : _base(std::move(l._base))
	{}

	// todo: muss nachher weg
	basic_strides(base_type const& l )
	    : _base(l)
	{}

	// todo: muss nachher weg
	basic_strides(base_type && l )
			: _base(std::move(l))
	{}

	~basic_strides() = default;

	basic_strides& operator=(basic_strides const& l ){
		_base = l._base;
		return *this;
	}

	basic_strides& operator=(basic_strides && l ){
		_base = std::move(l._base);
		return *this;
	}

	const_reference operator[] (size_type p) const{
		return _base[p];
	}

	const_pointer data() const{
		return _base.data();
	}

	const_reference at (size_type p) const{
		return _base.at(p);
	}


	bool empty() const{
		return _base.empty();
	}

	size_type size() const{
		return _base.size();
	}

	bool operator == (basic_strides const& b) const{
		return b._base == _base;
	}

	bool operator != (basic_strides const& b) const{
		return b._base != _base;
	}

	const_iterator begin() const{
		return _base.begin();
	}

	const_iterator end() const{
		return _base.end();
	}

	void clear() {
		this->_base.clear();
	}

	base_type const& base() const{
		return this->_base;
	}


protected:
	base_type _base;
};


using strides = basic_strides<std::size_t>;

}}}

#endif
