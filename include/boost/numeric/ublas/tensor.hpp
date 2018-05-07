//
//  Copyright (c) 2018
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB in producing this work.
//
//  And we acknowledge the support from all contributors.

/// \file tensor.hpp Definition for the class vector and its derivative

#ifndef _BOOST_UBLAS_TENSOR_
#define _BOOST_UBLAS_TENSOR_

#include <boost/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/storage.hpp>

#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>

//#include <boost/numeric/ublas/vector_expression.hpp>
//#include <boost/numeric/ublas/detail/vector_assign.hpp>
//#include <boost/serialization/collection_size_type.hpp>
//#include <boost/serialization/nvp.hpp>

#include <initializer_list>

#ifdef BOOST_UBLAS_CPP_GE_2011
#include <array>
#include <initializer_list>
#if defined(BOOST_MSVC) // For std::forward in fixed_vector
#include <utility>
#endif
#endif

/** \brief Base class for Vector container models
 *
 * it does not model the Vector concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Vector classes
 * We implement the casts to the statically derived type.
 */



namespace boost { namespace numeric { namespace ublas {

template<class T, class F, class A>
class tensor;


//TODO: put in fwd.hpp
struct tensor_tag {};


//TODO: put in expression_types.hpp

/** \brief Base class for Tensor Expression models
 *
 * it does not model the Tensor Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Tensor Expression classes.
 * We implement the casts to the statically derived type.
 */
template<class E>
class tensor_expression:
		public ublas_expression<E> {
public:
		static const unsigned complexity = 0;
		typedef E expression_type;
		typedef tensor_tag type_category;
		/* E can be an incomplete type - to define the following we would need more template arguments
		typedef typename E::size_type size_type;
		*/

		BOOST_UBLAS_INLINE
		const expression_type &operator () () const {
				return *static_cast<const expression_type *> (this);
		}
		BOOST_UBLAS_INLINE
		expression_type &operator () () {
				return *static_cast<expression_type *> (this);
		}

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
private:
		// projection types
		typedef tensor_range<E> tensor_range_type;
		typedef tensor_range<const E> const_tensor_range_type;
		typedef tensor_slice<E> tensor_slice_type;
		typedef tensor_slice<const E> const_tensor_slice_type;
		// tensor_indirect_type will depend on the A template parameter
		typedef basic_range<> default_range;    // required to avoid range/slice name confusion
		typedef basic_slice<> default_slice;
public:
		BOOST_UBLAS_INLINE
		const_tensor_range_type operator () (const default_range &r) const {
				return const_tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		tensor_range_type operator () (const default_range &r) {
				return tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		const_tensor_slice_type operator () (const default_slice &s) const {
				return const_tensor_slice_type (operator () (), s);
		}
		BOOST_UBLAS_INLINE
		tensor_slice_type operator () (const default_slice &s) {
				return tensor_slice_type (operator () (), s);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		const tensor_indirect<const E, indirect_array<A> > operator () (const indirect_array<A> &ia) const {
				return tensor_indirect<const E, indirect_array<A> >  (operator () (), ia);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		tensor_indirect<E, indirect_array<A> > operator () (const indirect_array<A> &ia) {
				return tensor_indirect<E, indirect_array<A> > (operator () (), ia);
		}

		BOOST_UBLAS_INLINE
		const_tensor_range_type project (const default_range &r) const {
				return const_tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		tensor_range_type project (const default_range &r) {
				return tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		const_tensor_slice_type project (const default_slice &s) const {
				return const_tensor_slice_type (operator () (), s);
		}
		BOOST_UBLAS_INLINE
		tensor_slice_type project (const default_slice &s) {
				return tensor_slice_type (operator () (), s);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		const tensor_indirect<const E, indirect_array<A> > project (const indirect_array<A> &ia) const {
				return tensor_indirect<const E, indirect_array<A> > (operator () (), ia);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		tensor_indirect<E, indirect_array<A> > project (const indirect_array<A> &ia) {
				return tensor_indirect<E, indirect_array<A> > (operator () (), ia);
		}
#endif
};


/** \brief output stream operator for tensor expressions
 *
 * Any vector expressions can be written to a standard output stream
 * as defined in the C++ standard library. For example:
 * \code
 * vector<float> v1(3),v2(3);
 * for(size_t i=0; i<3; i++)
 * {
 *       v1(i) = i+0.2;
 *       v2(i) = i+0.3;
 * }
 * cout << v1+v2 << endl;
 * \endcode
 * will display the some of the 2 vectors like this:
 * \code
 * [3](0.5,2.5,4.5)
 * \endcode
 *
 * \param os is a standard basic output stream
 * \param v is a vector expression
 * \return a reference to the resulting output stream
 */
template<class E, class T, class VE>
// BOOST_UBLAS_INLINE This function seems to be big. So we do not let the compiler inline it.
std::basic_ostream<E, T>&
operator << (std::basic_ostream<E, T> &os, const tensor_expression<VE> &v) {
		typedef typename VE::size_type size_type;
		size_type size = v ().size ();
		std::basic_ostringstream<E, T, std::allocator<E> > s;
		s.flags (os.flags ());
		s.imbue (os.getloc ());
		s.precision (os.precision ());
		s << '[' << size << "](";
		if (size > 0)
				s << v () (0);
		for (size_type i = 1; i < size; ++ i)
				s << ',' << v () (i);
		s << ')';
		return os << s.str ().c_str ();
}




/** \brief Base class for Tensor container models
 *
 * it does not model the Tensor concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Tensor classes
 * We implement the casts to the statically derived type.
 */
template<class C>
class tensor_container:
		public tensor_expression<C>
{
public:
		static const unsigned complexity = 0;
		typedef C container_type;
		typedef tensor_tag type_category;

		BOOST_UBLAS_INLINE
		const container_type &operator () () const {
				return *static_cast<const container_type *> (this);
		}
		BOOST_UBLAS_INLINE
		container_type &operator () () {
				return *static_cast<container_type *> (this);
		}

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
		using tensor_expression<C>::operator ();
#endif
};

/** @brief A dense tensor of values of type \c T.
		*
		* For a \f$n\f$-dimensional tensor \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
		* to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
		* Elements are constructed by \c A, which need not initialise their value.
		*
		* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
		* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
		*/
template<class T, class F = first_order, class A = unbounded_array<T,std::allocator<T>> >
class tensor:
		public tensor_container<tensor<T, F, A> >
{

	typedef tensor<T, F, A> self_type;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using tensor_container<self_type>::operator ();
#endif

	typedef typename A::size_type size_type;
	typedef typename A::difference_type difference_type;
	typedef T value_type;
	typedef typename type_traits<T>::const_reference const_reference;
	typedef T &reference;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef A array_type;
	typedef F layout_type;
//	typedef const tensor_reference<const self_type> const_closure_type;
//	typedef tensor_reference<self_type> closure_type;
	typedef self_type tensor_temporary_type;
	typedef dense_tag storage_category;

	// Reverse iterator
//	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
//	typedef reverse_iterator_base<iterator> reverse_iterator;

	/** @brief Standard constructor of the tensor template class
	 *
	 * By default it is empty, i.e. \c size()==0.
	 */
	BOOST_UBLAS_INLINE
	constexpr tensor ()
		: tensor_container<self_type>()
		, extents_()
		, strides_()
		, data_()
	{
	}


	/** @brief Constructor of the tensor template class with a predefined size
	 *
	 * Layout or storage format is automatically set to first-order.
	 * By default, its elements are initialized to 0.
	 *
	 * @code tensor A{4,2,3}; @endcode
	 *
	 * @param l initializer list for setting the dimension extents of the tensor
	 */
	explicit BOOST_UBLAS_INLINE
	tensor (std::initializer_list<size_type> l)
		: tensor_container<self_type>()
		, extents_(std::move(l))
		, strides_(extents_)
		, data_(extents_.product())
	{
	}


	/** @brief Constructor of the tensor template class
		*
		* Layout or storage format is automatically set to first-order.
		* By default, its elements are initialized to 0.
		*
		* @code tensor A{extents{4,2,3}}; @endcode
		*
		* @param e initial tensor dimension extents
		*/
	explicit BOOST_UBLAS_INLINE
	tensor (extents const& e)
		: tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product())
	{}


	/** @brief Constructor of the tensor template class
	 *
	 *  @code tensor A{extents{4,2,3},vector}; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param data container of \c array_type
	 */
	BOOST_UBLAS_INLINE
	tensor (extents const& e, const array_type &data)
		: tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (data)
	{
		if(this->extents_.product() != this->data_.size())
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: size of provided data and specified extents do not match.");
	}



	/** @brief Constructor of the tensor template class
	 *
	 *  @param e initial tensor dimension extents
	 *  @param i container of \c array_type
	 */
	BOOST_UBLAS_INLINE
	tensor (extents const& e, const value_type &i)
		: tensor_container<self_type> ()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product(), i)
	{}


#if 0
	/// \brief Copy-constructor of a tensor
	/// \param v is the tensor to be duplicated
	BOOST_UBLAS_INLINE
	tensor (const tensor &v):
		tensor_container<self_type> (),
		data_ (v.data_)
	{}

	/// \brief Copy-constructor of a tensor from a tensor_expression
	/// Depending on the tensor_expression, this constructor can have the cost of the computations
	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
	/// \param ae the tensor_expression which values will be duplicated into the tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor (const tensor_expression<AE> &ae):
		tensor_container<self_type> (),
		data_ (ae ().size ())
	{
		// tensor_assign<scalar_assign> (*this, ae);
	}
#endif
	// -----------------------
	// Random Access Container
	// -----------------------

	/// \brief Return true if the tensor is empty (\c size==0)
	/// \return \c true if empty, \c false otherwise
	BOOST_UBLAS_INLINE
	bool empty () const {
		return this->data_.empty();
	}

	// ---------
	// Accessors
	// ---------

	/// \brief Return the size of the tensor
	BOOST_UBLAS_INLINE
	size_type size () const {
		return this->data_.size ();
	}

	/// \brief Return the size of the tensor
	BOOST_UBLAS_INLINE
	size_type rank () const {
		return this->extents_.size();
	}

	// -----------------
	// Storage accessors
	// -----------------

	/// \brief Return a \c const reference to the container. Useful to access data directly for specific type of container.
	BOOST_UBLAS_INLINE
	const_pointer data () const {
		return this->data_.begin();
	}

	/// \brief Return a reference to the container. Useful to speed-up write operations to the data in very specific case.
	BOOST_UBLAS_INLINE
	pointer data () {
		return this->data_.end();
	}


	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[i]
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	const_reference operator [] (size_type i) const {
		return this->data_[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[i]
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	reference operator [] (size_type i)
	{
		return this->data_[i];
	}


	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[i]
	/// \param i index of the element

	template<class ... size_types>
	BOOST_UBLAS_INLINE
	const_reference at (size_type i, size_types ... is) const
	{
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[ access<0>(0,i,is...)];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[i]
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	template<class ... size_types>
	reference at (size_type i, size_types ... is)
	{
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[ access<0>(0,i,is...)];
	}

#if 0
	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	///
	///
	template<class ... size_types>
	BOOST_UBLAS_INLINE
	const_reference operator () (size_type i, size_types ... sizes) const
	{
		return access<0>(i,std::forward<size_type>(sizes)...);

//		return this->data_(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	reference operator [] (size_type i) {
		return (*this) (i);
	}

	// ------------------
	// Element assignment
	// ------------------

	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	// XXX semantic of this is to insert a new element and therefore size=size+1 ?
	BOOST_UBLAS_INLINE
	reference insert_element (size_type i, const_reference t) {
		return (data () [i] = t);
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	void erase_element (size_type i) {
		data () [i] = value_type/*zero*/();
	}

	// -------
	// Zeroing
	// -------

	/// \brief Clear the tensor, i.e. set all values to the \c zero value.
	BOOST_UBLAS_INLINE
	void clear () {
		std::fill (data ().begin (), data ().end (), value_type/*zero*/());
	}

	// --------
	// Resizing
	// --------


	/// \brief Resize the tensor
	/// Resize the tensor to a new size. If \c preserve is true, data are copied otherwise data are lost. If the new size is bigger, the remaining values are filled in with the initial value (0 by default) in the case of \c unbounded_array, which is the container by default. If the new size is smaller, last values are lost. This behaviour can be different if you explicitely specify another type of container.
	/// \param size new size of the tensor
	/// \param preserve if true, keep values
	BOOST_UBLAS_INLINE
	void resize (size_type size, bool preserve = true) {
		if (preserve)
			data ().resize (size, typename A::value_type ());
		else
			data ().resize (size);
	}

	// Assignment
#ifdef BOOST_UBLAS_MOVE_SEMANTICS

	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// \param v is the source tensor
	/// \return a reference to a tensor (i.e. the destination tensor)
	/*! @note "pass by value" the key idea to enable move semantics */
	BOOST_UBLAS_INLINE
	tensor &operator = (tensor v) {
		assign_temporary(v);
		return *this;
	}
#else
	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// \param v is the source tensor
	/// \return a reference to a tensor (i.e. the destination tensor)
	BOOST_UBLAS_INLINE
	tensor &operator = (const tensor &v) {
		data () = v.data ();
		return *this;
	}
#endif

	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor). This method does not create any temporary.
	/// \param v is the source tensor container
	/// \return a reference to a tensor (i.e. the destination tensor)
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator = (const tensor_container<C> &v) {
		resize (v ().size (), false);
		assign (v);
		return *this;
	}

	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// \param v is the source tensor
	/// \return a reference to a tensor (i.e. the destination tensor)
	BOOST_UBLAS_INLINE
	tensor &assign_temporary (tensor &v) {
		swap (v);
		return *this;
	}

	/// \brief Assign the result of a tensor_expression to the tensor
	/// Assign the result of a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator = (const tensor_expression<AE> &ae)
	{
		self_type temporary (ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the result of a tensor_expression to the tensor
	/// Assign the result of a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_assign> (*this, ae);
		return *this;
	}

	// -------------------
	// Computed assignment
	// -------------------

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator += (const tensor_expression<AE> &ae) {
		self_type temporary (*this + ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator += (const tensor_container<C> &v) {
		plus_assign (v);
		return *this;
	}

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &plus_assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator -= (const tensor_expression<AE> &ae) {
		self_type temporary (*this - ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator -= (const tensor_container<C> &v) {
		minus_assign (v);
		return *this;
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &minus_assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the product of the tensor and a scalar to the tensor
	/// Assign the product of the tensor and a scalar to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting tensor
	template<class AT>
	BOOST_UBLAS_INLINE
	tensor &operator *= (const AT &at)
	{
//		tensor_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}

	/// \brief Assign the division of the tensor by a scalar to the tensor
	/// Assign the division of the tensor by a scalar to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting tensor
	template<class AT>
	BOOST_UBLAS_INLINE
	tensor &operator /= (const AT &at)
	{
//		tensor_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}

	// --------
	// Swapping
	// --------

	/// \brief Swap the content of the tensor with another tensor
	/// \param v is the tensor to be swapped with
	BOOST_UBLAS_INLINE
	void swap (tensor &v) {
		if (this != &v) {
			data ().swap (v.data ());
		}
	}

	/// \brief Swap the content of two tensors
	/// \param v1 is the first tensor. It takes values from v2
	/// \param v2 is the second tensor It takes values from v1
	BOOST_UBLAS_INLINE
	friend void swap (tensor &v1, tensor &v2) {
		v1.swap (v2);
	}

	// Iterator types
private:
	// Use the storage array iterator
	typedef typename A::const_iterator const_subiterator_type;
	typedef typename A::iterator subiterator_type;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_iterator<self_type, dense_random_access_iterator_tag> iterator;
	typedef indexed_const_iterator<self_type, dense_random_access_iterator_tag> const_iterator;
#else
	class const_iterator;
	class iterator;
#endif

	// --------------
	// Element lookup
	// --------------

	/// \brief Return a const iterator to the element \e i
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	const_iterator find (size_type i) const {
#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator (*this, data ().begin () + i);
#else
		return const_iterator (*this, i);
#endif
	}

	/// \brief Return an iterator to the element \e i
	/// \param i index of the element
	BOOST_UBLAS_INLINE
	iterator find (size_type i) {
#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return iterator (*this, data ().begin () + i);
#else
		return iterator (*this, i);
#endif
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator:
			public container_const_reference<tensor>,
			public random_access_iterator_base<dense_random_access_iterator_tag,
			const_iterator, value_type, difference_type> {
	public:
		typedef typename tensor::difference_type difference_type;
		typedef typename tensor::value_type value_type;
		typedef typename tensor::const_reference reference;
		typedef const typename tensor::pointer pointer;

		// ----------------------------
		// Construction and destruction
		// ----------------------------


		BOOST_UBLAS_INLINE
		const_iterator ():
			container_const_reference<self_type> (), it_ () {}
		BOOST_UBLAS_INLINE
		const_iterator (const self_type &v, const const_subiterator_type &it):
			container_const_reference<self_type> (v), it_ (it) {}
		BOOST_UBLAS_INLINE
		const_iterator (const typename self_type::iterator &it):  // ISSUE tensor:: stops VC8 using std::iterator here
			container_const_reference<self_type> (it ()), it_ (it.it_) {}

		// ----------
		// Arithmetic
		// ----------

		/// \brief Increment by 1 the position of the iterator
		/// \return a reference to the const iterator
		BOOST_UBLAS_INLINE
		const_iterator &operator ++ () {
			++ it_;
			return *this;
		}

		/// \brief Decrement by 1 the position of the iterator
		/// \return a reference to the const iterator
		BOOST_UBLAS_INLINE
		const_iterator &operator -- () {
			-- it_;
			return *this;
		}

		/// \brief Increment by \e n the position of the iterator
		/// \return a reference to the const iterator
		BOOST_UBLAS_INLINE
		const_iterator &operator += (difference_type n) {
			it_ += n;
			return *this;
		}

		/// \brief Decrement by \e n the position of the iterator
		/// \return a reference to the const iterator
		BOOST_UBLAS_INLINE
		const_iterator &operator -= (difference_type n) {
			it_ -= n;
			return *this;
		}

		/// \brief Return the different in number of positions between 2 iterators
		BOOST_UBLAS_INLINE
		difference_type operator - (const const_iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ - it.it_;
		}

		/// \brief Dereference an iterator
		/// Dereference an iterator: a bounds' check is done before returning the value. A bad_index() expection is returned if out of bounds.
		/// \return a const reference to the value pointed by the iterator
		BOOST_UBLAS_INLINE
		const_reference operator * () const {
			BOOST_UBLAS_CHECK (it_ >= (*this) ().begin ().it_ && it_ < (*this) ().end ().it_, bad_index ());
			return *it_;
		}

		/// \brief Dereference an iterator at the n-th forward value
		/// Dereference an iterator at the n-th forward value, that is the value pointed by iterator+n.
		/// A bounds' check is done before returning the value. A bad_index() expection is returned if out of bounds.
		/// \return a const reference
		BOOST_UBLAS_INLINE
		const_reference operator [] (difference_type n) const {
			return *(it_ + n);
		}

		// Index
		/// \brief return the index of the element referenced by the iterator
		BOOST_UBLAS_INLINE
		size_type index () const {
			BOOST_UBLAS_CHECK (it_ >= (*this) ().begin ().it_ && it_ < (*this) ().end ().it_, bad_index ());
			return it_ - (*this) ().begin ().it_;
		}

		// Assignment
		BOOST_UBLAS_INLINE
		/// \brief assign the value of an iterator to the iterator
		const_iterator &operator = (const const_iterator &it) {
			container_const_reference<self_type>::assign (&it ());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		/// \brief compare the value of two itetarors
		/// \return true if they reference the same element
		BOOST_UBLAS_INLINE
		bool operator == (const const_iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ == it.it_;
		}


		/// \brief compare the value of two iterators
		/// \return return true if the left-hand-side iterator refers to a value placed before the right-hand-side iterator
		BOOST_UBLAS_INLINE
		bool operator < (const const_iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ < it.it_;
		}

	private:
		const_subiterator_type it_;

		friend class iterator;
	};
#endif

	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator begin () const {
		return find (0);
	}

	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cbegin () const {
		return begin ();
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator end () const {
		return find (data_.size ());
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cend () const {
		return end ();
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class iterator:
			public container_reference<tensor>,
			public random_access_iterator_base<dense_random_access_iterator_tag,
			iterator, value_type, difference_type> {
	public:
		typedef typename tensor::difference_type difference_type;
		typedef typename tensor::value_type value_type;
		typedef typename tensor::reference reference;
		typedef typename tensor::pointer pointer;


		// Construction and destruction
		BOOST_UBLAS_INLINE
		iterator ():
			container_reference<self_type> (), it_ () {}
		BOOST_UBLAS_INLINE
		iterator (self_type &v, const subiterator_type &it):
			container_reference<self_type> (v), it_ (it) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		iterator &operator ++ () {
			++ it_;
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator &operator -- () {
			-- it_;
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator &operator += (difference_type n) {
			it_ += n;
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator &operator -= (difference_type n) {
			it_ -= n;
			return *this;
		}
		BOOST_UBLAS_INLINE
		difference_type operator - (const iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ - it.it_;
		}

		// Dereference
		BOOST_UBLAS_INLINE
		reference operator * () const {
			BOOST_UBLAS_CHECK (it_ >= (*this) ().begin ().it_ && it_ < (*this) ().end ().it_ , bad_index ());
			return *it_;
		}
		BOOST_UBLAS_INLINE
		reference operator [] (difference_type n) const {
			return *(it_ + n);
		}

		// Index
		BOOST_UBLAS_INLINE
		size_type index () const {
			BOOST_UBLAS_CHECK (it_ >= (*this) ().begin ().it_ && it_ < (*this) ().end ().it_ , bad_index ());
			return it_ - (*this) ().begin ().it_;
		}

		// Assignment
		BOOST_UBLAS_INLINE
		iterator &operator = (const iterator &it) {
			container_reference<self_type>::assign (&it ());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ == it.it_;
		}
		BOOST_UBLAS_INLINE
		bool operator < (const iterator &it) const {
			BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
			return it_ < it.it_;
		}

	private:
		subiterator_type it_;

		friend class const_iterator;
	};
#endif

	/// \brief Return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	iterator begin () {
		return find (0);
	}

	/// \brief Return an iterator at the end of the tensor
	BOOST_UBLAS_INLINE
	iterator end () {
		return find (data_.size ());
	}



	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rbegin () const {
		return const_reverse_iterator (end ());
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crbegin () const {
		return rbegin ();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rend () const {
		return const_reverse_iterator (begin ());
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crend () const {
		return rend ();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rbegin () {
		return reverse_iterator (end ());
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rend () {
		return reverse_iterator (begin ());
	}

	// -------------
	// Serialization
	// -------------

	/// Serialize a tensor into and archive as defined in Boost
	/// \param ar Archive object. Can be a flat file, an XML file or any other stream
	/// \param file_version Optional file version (not yet used)
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */){
		ar & serialization::make_nvp("data",data_);
	}
#endif	

private:

	/** \brief Access function with multi-indices
	 *
	 * \param i multi-index vector of length p
	 * \param w stride vector of length p
	*/
	BOOST_UBLAS_INLINE
	size_type access(std::vector<size_type> const& i)
	{
		const auto p = this->rank();
		size_type sum = 0u;
		for(auto r = 0u; r < p; ++r)
			sum += i[r]*strides_[r];
		return sum;
	}

	BOOST_UBLAS_INLINE
	template<std::size_t r, class ... size_types>
	size_type access(size_type sum, size_type i, size_types ... is)
	{
		sum+=i*strides_[r];
		if constexpr (sizeof...(is) == 0)
			return sum;
		else
			return access<r+1>(sum,std::forward<size_type>(is)...);
	}

	extents extents_;	
	strides<layout_type> strides_;
	array_type data_;
};

}}} // namespaces



#endif
