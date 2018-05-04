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


#ifndef _BOOST_UBLAS_LAYOUT_
#define _BOOST_UBLAS_LAYOUT_

#include <vector>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <initializer_list>

namespace boost { namespace numeric { namespace ublas {

template<class __int_type, class __allocator_type = std::allocator<__int_type>>
class basic_layout
{
	using base_type = std::vector<__int_type, __allocator_type>;
	static_assert( std::numeric_limits<typename base_type::value_type>::is_integer, "Static error in basic_layout: type must be of type integer.");
	static_assert(!std::numeric_limits<typename base_type::value_type>::is_signed,  "Static error in basic_layout: type must be of type unsigned integer.");

public:
	using value_type = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using size_type = typename base_type::size_type;
	using const_pointer = typename base_type::const_pointer;
	using const_iterator = typename base_type::const_iterator;

	constexpr explicit basic_layout() = default;

	explicit basic_layout(std::initializer_list<value_type> l)
	    : _base(l)
	{
		if (!this->valid())
			throw std::length_error("Error in basic_layout::basic_layout() : layout tuple is not a valid permutation.");
	}

	basic_layout(base_type const& l )
	    : _base(l)
	{
		if (!this->valid())
			throw std::length_error("Error in basic_layout::basic_layout() : layout tuple is not a valid permutation.");
	}

	basic_layout(base_type && l )
	    : _base(std::move(l))
	{
		if (!this->valid())
			throw std::length_error("Error in basic_layout::basic_layout() : layout tuple is not a valid permutation.");
	}


	basic_layout(basic_layout const& l )
	    : _base(l._base)
	{		
	}

	basic_layout(basic_layout && l )
	    : _base(std::move(l._base))
	{
	}

	basic_layout& operator=(basic_layout const& l )
	{
		_base.operator =(l._base);
		return *this;
	}

	basic_layout& operator=(basic_layout && l )
	{
		_base.operator =(std::move(l._base));
		return *this;
	}

	basic_layout
	inverse() const
	{
		base_type v(this->size());

		for(auto r = 1ul; r <= this->size(); ++r)
			v[_base[r-1]-1] = r;

		return v;
	}

	static basic_layout
	first_order(std::size_t rank)
	{
		base_type l(rank);
		std::iota(l.begin(), l.end(), value_type(1));
		return basic_layout(l);
	}

	static basic_layout
	last_order(std::size_t rank)
	{
		base_type l(rank);
		std::iota(l.rbegin(), l.rend(), value_type(1));
		return basic_layout(l);
	}

	bool valid() const
	{
		const auto rank = this->size();
		std::vector<bool> already_indexed(rank, false);
		for(auto index : *this)
		{
			if(index == 0 || index > rank || already_indexed[index-1])
				return false;
			already_indexed[index-1] = true;
		}
		return true;
	}

	const_reference back() const
	{
		return _base.back();
	}
	const_reference front() const
	{
		return _base.front();
	}

	const_pointer data() const
	{
		return &_base[0];
	}

	const_reference operator[] (size_type p) const
	{
		return _base[p];
	}

	const_reference at (size_type p) const
	{
		return _base.at(p);
	}


	bool empty() const
	{
		return _base.empty();
	}

	size_type size() const
	{
		return _base.size();
	}

	bool operator == (basic_layout const& b) const
	{
		return _base == b._base;
	}

	bool operator != (basic_layout const& b) const
	{
		return _base != b._base;
	}

	bool operator == (basic_layout && b)
	{
		return _base == b._base;
	}

	bool operator != (basic_layout && b)
	{
		return _base != b._base;
	}

	bool operator == (base_type const& b) const
	{
		return _base == b;
	}

	bool operator != (base_type const& b) const
	{
		return _base != b;
	}

	const_iterator begin() const
	{
		return _base.begin();
	}

	const_iterator end() const
	{
		return _base.end();
	}

	base_type const& base() const { return _base; }

private:
	base_type _base;
};

using layout = basic_layout<std::size_t>;

}}}


#endif
