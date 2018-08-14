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


#ifndef _BOOST_UBLAS_TENSOR_SPAN_
#define _BOOST_UBLAS_TENSOR_SPAN_

#include <typeinfo>
#include <limits>

#include "../detail/config.hpp"


namespace boost { namespace numeric { namespace ublas {

/** \class span
	* \ingroup Core_Module
	*
	* \brief Selection operator class to initialize stl::multi_subarray
	*
	* This class is used to generate stl::multi_subarray from stl::multi_array and to
	* work on views.
	* \note zero based indexing is used.
	*
	*/


struct sliced_tag {};
struct strided_tag {};

static constexpr inline std::size_t end = std::numeric_limits<std::size_t>::max()-std::size_t(1);
static constexpr inline std::size_t all = std::numeric_limits<std::size_t>::max();

//using offsets = std::vector<std::ptrdiff_t>;

template<class span_tag, class unsigned_type>
class span;


template<>
class span<strided_tag, std::size_t>
{
public:
	using span_tag = strided_tag;
	using value_type = std::size_t;
	constexpr explicit span() : first_{}, last_{}, step_{}, size_{} {}

	span(value_type f, value_type s, value_type l)
		: first_(f)
		, last_ (l)
		, step_ (s)
	{
		if(f == l){
			last_ = l;
			size_ = value_type(1);
		}
		else {
			if(s == 0 && f != l)
				throw std::runtime_error("Error in span::span : cannot have a step_ equal to zero.");

			if(f > l)
				throw std::runtime_error("Error in span::span: last_ is smaller than first");

			last_ = l - ((l-f)%s);
			size_ = (last_-first_)/s+value_type(1);
		}
	}

	span(span const& other)
		: first_(other.first_)
		, last_ (other.last_ )
		, step_ (other.step_ )
		, size_ (other.size_ )
	{
	}

	span& operator=(span const& other)
	{
		first_ = other.first_;
		last_  = other.last_ ;
		step_  = other.step_ ;
		size_  = other.size_ ;
		return *this;
	}

	BOOST_UBLAS_INLINE auto first() const {return first_; }
	BOOST_UBLAS_INLINE auto last () const {return last_ ; }
	BOOST_UBLAS_INLINE auto step () const {return step_ ; }
	BOOST_UBLAS_INLINE auto size () const {return size_ ; }

	~span() = default;

	BOOST_UBLAS_INLINE
	value_type operator[] (std::size_t idx) const
	{
		return first_ + idx * step_;
	}

	BOOST_UBLAS_INLINE
	span operator()(const span &rhs) const
	{
		auto const& lhs = *this;
		return span(
					rhs.first_*lhs.step_ + lhs.first_,
					lhs.step_ *rhs.step_,
					rhs.last_ *lhs.step_ + lhs.first_ );
	}

protected:

	value_type first_, last_ , step_, size_;
};

using strided_span = span<strided_tag, std::size_t>;


/////////////


template<>
class span<sliced_tag, std::size_t> :
		private span<strided_tag, std::size_t>
{
	using super_type = span<strided_tag,std::size_t>;
public:
	using span_tag = sliced_tag;
	using value_type = typename super_type::value_type;
	constexpr explicit span() = default;

	span(value_type f, value_type l)
		: super_type(f, value_type(1), l )
	{
	}

	span(span const& other)
		: super_type(other)
	{
	}

	BOOST_UBLAS_INLINE
	span& operator=(const span &other)
	{
		super_type::operator=(other);
		return *this;
	}

	~span() = default;

	BOOST_UBLAS_INLINE
	value_type operator[] (std::size_t idx) const
	{
		return super_type::operator [](idx);
	}

	BOOST_UBLAS_INLINE auto first() const {return super_type::first(); }
	BOOST_UBLAS_INLINE auto last () const {return super_type::last (); }
	BOOST_UBLAS_INLINE auto step () const {return super_type::step (); }
	BOOST_UBLAS_INLINE auto size () const {return super_type::size (); }

	BOOST_UBLAS_INLINE
	span operator()(const span &rhs) const
	{
		auto const& lhs = *this;
		return span( rhs.first_ + lhs.first_, rhs.last_  + lhs.first_ );
	}
};


using sliced_span = span<sliced_tag, std::size_t>;

BOOST_UBLAS_INLINE
template<class unsigned_type>
auto ran(unsigned_type f, unsigned_type l)
{
	return sliced_span(f,l);
}

BOOST_UBLAS_INLINE
template<class unsigned_type>
auto ran(unsigned_type f, unsigned_type s, unsigned_type l)
{
	return strided_span(f,s,l);
}
}
}
}


template<class span_tag_lhs, class span_tag_rhs, class unsigned_type>
bool operator==(
		boost::numeric::ublas::span<span_tag_lhs,unsigned_type> const& lhs,
		boost::numeric::ublas::span<span_tag_rhs,unsigned_type> const& rhs)
{
	return lhs.first() == rhs.first() && lhs.last() == rhs.last() && lhs.step() == rhs.step();
}

#endif // FHG_range_H
