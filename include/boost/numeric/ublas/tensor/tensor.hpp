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

/// \file tensor.hpp Definition for the tensor template class

#ifndef _BOOST_UBLAS_TENSOR_IMPL_
#define _BOOST_UBLAS_TENSOR_IMPL_

#include <boost/config.hpp>
#include <boost/numeric/ublas/storage.hpp>

#include <initializer_list>

#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "extents.hpp"
#include "strides.hpp"



namespace boost { namespace numeric { namespace ublas {

template<class T, class F, class A>
class tensor;

///** \brief Base class for Tensor container models
// *
// * it does not model the Tensor concept but all derived types should.
// * The class defines a common base type and some common interface for all
// * statically derived Tensor classes
// * We implement the casts to the statically derived type.
// */
//template<class C>
//class tensor_container:
//		public detail::tensor_expression<C>
//{
//public:
//	static const unsigned complexity = 0;
//	typedef C container_type;
//	typedef tensor_tag type_category;

//	BOOST_UBLAS_INLINE
//	const container_type &operator () () const {
//			return *static_cast<const container_type *> (this);
//	}
//	BOOST_UBLAS_INLINE
//	container_type &operator () () {
//			return *static_cast<container_type *> (this);
//	}
//};

/** @brief A dense tensor of values of type \c T.
		*
		* For a \f$n\f$-dimensional tensor \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
		* to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
		* Elements are constructed by \c A, which need not initialise their value.
		*
		* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
		* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
		*/
template<class T, class F = first_order, class A = std::vector<T,std::allocator<T>> >
class tensor:
		public detail::tensor_expression<tensor<T, F, A>,tensor<T, F, A>>
{

	static_assert( std::is_same_v<F,first_order> ||
								 std::is_same_v<F,last_order >, "boost::numeric::tensor template class only supports first- or last-order storage formats.");

	using self_type  = tensor<T, F, A>;
public:



	template<class derived_type>
	using tensor_expression_type = detail::tensor_expression<self_type,derived_type>;

	template<class derived_type>
	using matrix_expression_type = matrix_expression<derived_type>;

	using super_type = tensor_expression_type<self_type>;

//	static_assert(std::is_same_v<tensor_expression_type<self_type>, detail::tensor_expression<tensor<T,F,A>,tensor<T,F,A>>>, "tensor_expression_type<self_type>");

	using array_type  = A;
	using layout_type = F;


	using size_type       = typename array_type::size_type;
	using difference_type = typename array_type::difference_type;
	using value_type      = typename array_type::value_type;

	using reference       = typename array_type::reference;
	using const_reference = typename array_type::const_reference;

	using pointer         = typename array_type::pointer;
	using const_pointer   = typename array_type::const_pointer;

	using iterator        = typename array_type::iterator;
	using const_iterator  = typename array_type::const_iterator;

	using reverse_iterator        = typename array_type::reverse_iterator;
	using const_reverse_iterator  = typename array_type::const_reverse_iterator;

	using tensor_temporary_type = self_type;
	using storage_category = dense_tag;

	using strides_type = basic_strides<std::size_t,layout_type>;
	using extents_type = shape;

	using matrix_type     = matrix<value_type,layout_type,array_type>;



	/** @brief Standard constructor of the tensor template class
	 *
	 * By default it is empty, i.e. \c size()==0.
	 */
	BOOST_UBLAS_INLINE
	constexpr tensor ()
		: tensor_expression_type<self_type>() // container_type
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
	 * @code tensor<float> A{4,2,3}; @endcode
	 *
	 * @param l initializer list for setting the dimension extents of the tensor
	 */
	explicit BOOST_UBLAS_INLINE
	tensor (std::initializer_list<size_type> l)
		: tensor_expression_type<self_type>()
		, extents_ (std::move(l))
		, strides_ (extents_)
		, data_    (extents_.product())
	{
	}


	/** @brief Constructor of the tensor template class
		*
		* Layout or storage format is automatically set to first-order.
		* By default, its elements are initialized to 0.
		*
		* @code tensor<float> A{extents{4,2,3}}; @endcode
		*
		* @param e initial tensor dimension extents
		*/
	explicit BOOST_UBLAS_INLINE
	tensor (shape const& e)
		: tensor_expression_type<self_type>() //tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product())
	{}


	/** @brief Constructor of the tensor template class
	 *
	 *
	 *  @code tensor<float> A{extents{4,2,3}, array }; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param data container of \c array_type
	 */
	BOOST_UBLAS_INLINE
	tensor (shape const& e, const array_type &data)
		: tensor_expression_type<self_type>() //tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (data)
	{
		if(this->extents_.product() != this->data_.size())
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: size of provided data and specified extents do not match.");
	}



	/** @brief Constructor of the tensor template class
	 *
	 *  @code tensor<float> A{extents{4,2,3}, 1 }; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param i initial value of all elements of type \c value_type
	 */
	BOOST_UBLAS_INLINE
	tensor (shape const& e, const value_type &i)
		: tensor_expression_type<self_type>() //tensor_container<self_type> ()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product(), i)
	{}



	/** @brief Copy Constructor of the tensor template class
	 *
	 *  @param v tensor to be copied.
	 */
	BOOST_UBLAS_INLINE
	tensor (const tensor &v)
		: tensor_expression_type<self_type>()
		, extents_ (v.extents_)
		, strides_ (v.strides_)
		, data_    (v.data_   )
	{}



	/** @brief Move Constructor of the tensor template class
	 *
	 *  @param v tensor to be moved.
	 */
	BOOST_UBLAS_INLINE
	tensor (tensor &&v)
		: tensor_expression_type<self_type>() //tensor_container<self_type> ()
		, extents_ (std::move(v.extents_))
		, strides_ (std::move(v.strides_))
		, data_    (std::move(v.data_   ))
	{}


	/** @brief Constructor of the tensor template class
	 *
	 *  @param v matrix to be copied.
	 */
	BOOST_UBLAS_INLINE
	tensor (const matrix_type &v)
		: tensor_expression_type<self_type>()
		, extents_ ()
		, strides_ ()
		, data_    (v.data())
	{
		if(v.size1()*v.size2() != 0){
			extents_ = extents_type{v.size1(),v.size2()};
			strides_ = strides_type(extents_);
		}
	}

	/** @brief Constructor of the tensor template class
	 *
	 *  @param v matrix to be moved.
	 */
	BOOST_UBLAS_INLINE
	tensor (matrix_type &&v)
		: tensor_expression_type<self_type>()
		, extents_ {}
		, strides_ {}
		, data_    (std::move(v.data()))
	{
		if(v.size1()*v.size2() != 0){
			extents_ = extents_type{v.size1(),v.size2()};
			strides_ = strides_type(extents_);
		}
	}


//	/** @brief Copy Constructor of the tensor template class
//	 *
//	 *  @param v tensor to be copied.
//	 */
//	BOOST_UBLAS_INLINE
//	template<class other_layout>
//	tensor (const tensor<value_type, layout_type> &v)
//		: tensor_container<self_type> ()
//		, extents_ (v.extents_)
//		, strides_ (v.strides_)
//		, data_    (v.data_   )
//	{}

	/// \brief Copy-constructor of a tensor from a tensor_expression
	/// Depending on the tensor_expression, this constructor can have the cost of the computations
	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
	/// \param ae the tensor_expression which values will be duplicated into the tensor
	BOOST_UBLAS_INLINE
	template<class derived_type>
	tensor (const tensor_expression_type<derived_type> &expr)
		: tensor_expression_type<self_type> ()
		, extents_ ( detail::retrieve_extents(expr) )
		, strides_ ( extents_ )
		, data_    ( extents_.product() )
	{
		static_assert( detail::has_tensor_types<self_type, tensor_expression_type<derived_type>>::value,
									 "Error in boost::numeric::ublas::tensor: expression does not contain a tensor. cannot retrieve shape.");
		detail::eval( *this, expr );
	}

//	/// Depending on the tensor_expression, this constructor can have the cost of the computations
//	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
//	/// \param ae the tensor_expression which values will be duplicated into the tensor
//	BOOST_UBLAS_INLINE
//	template<class derived_type>
//	tensor (const matrix_expression_type<derived_type> &expr)
//		: tensor_expression_type<self_type> ()
//		, extents_ (  )
//		, strides_ ( extents_ )
//		, data_    ( extents_.product() )
//	{
//		static_assert( detail::has_tensor_types<self_type, tensor_expression_type<derived_type>>::value,
//									 "Error in boost::numeric::ublas::tensor: expression does not contain a tensor. cannot retrieve shape.");
//		detail::eval( *this, expr );
//	}

	/** @brief Evaluates the tensor_expression and assigns the results to the tensor
	 *
	 * @code A = B + C * 2;  @endcode
	 *
	 * @note rank and dimension extents of the tensors in the expressions must conform with this tensor.
	 *
	 * @param expr expression that is evaluated.
	 */
	BOOST_UBLAS_INLINE
	template<class derived_type>
	tensor &operator = (const tensor_expression_type<derived_type> &expr)
	{
		detail::eval(*this, expr);
		return *this;
	}

	tensor& operator=(tensor other)
	{
		swap (*this, other);
		return *this;
	}


	/** @brief Returns true if the tensor is empty (\c size==0) */
	BOOST_UBLAS_INLINE
	bool empty () const {
		return this->data_.empty();
	}


	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size () const {
		return this->data_.size ();
	}

	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size (size_type r) const {
		return this->extents_.at(r);
	}

	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type rank () const {
		return this->extents_.size();
	}

	/** @brief Returns the strides of the tensor */
	BOOST_UBLAS_INLINE
	strides_type const& strides () const {
		return this->strides_;
	}

	/** @brief Returns the extents of the tensor */
	BOOST_UBLAS_INLINE
	extents_type const& extents () const {
		return this->extents_;
	}


	/** @brief Returns a \c const reference to the container. */
	BOOST_UBLAS_INLINE
	const_pointer data () const {
		return this->data_.data();
	}

	/** @brief Returns a \c const reference to the container. */
	BOOST_UBLAS_INLINE
	pointer data () {
		return this->data_.data();
	}

	/** @brief Element access using a single index.
	 *
	 *  @code auto a = A[i]; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	const_reference operator [] (size_type i) const {
		return this->data_[i];
	}

	/** @brief Element access using a single index.
	 *
	 *
	 *  @code A[i] = a; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	reference operator [] (size_type i)
	{
		return this->data_[i];
	}


	/** @brief Element access using a multi-index or single-index.
	 *
	 *
	 *  @code auto a = A.at(i,j,k); @endcode or
	 *  @code auto a = A.at(i);     @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
	 *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
	 */
	template<class ... size_types>
	BOOST_UBLAS_INLINE
	const_reference at (size_type i, size_types ... is) const {
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[detail::access<0>(0ul,this->strides_,i,is...)];
	}

	/** @brief Element access using a multi-index or single-index.
	 *
	 *
	 *  @code A.at(i,j,k) = a; @endcode or
	 *  @code A.at(i) = a;     @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
	 *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
	 */
	BOOST_UBLAS_INLINE
	template<class ... size_types>
	reference at (size_type i, size_types ... is) {
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[detail::access<0ul>(0ul,this->strides_,i,is...)];
	}


	/** @brief Element access using a single index.
	 *
	 *
	 *  @code A(i) = a; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	const_reference operator()(size_type i) const {
		return this->data_[i];
	}


	/** @brief Element access using a single index.
	 *
	 *
	 *  @code A(i) = a; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	reference operator()(size_type i){
		return this->data_[i];
	}

	/** @brief Reshapes the tensor
	 *
	 *
	 * (1) @code A.reshape(extents{m,n,o});     @endcode or
	 * (2) @code A.reshape(extents{m,n,o},4);   @endcode
	 *
	 * If the size of this smaller than the specified extents than
	 * default constructed (1) or specified (2) value is appended.
	 *
	 * @note rank of the tensor might also change.
	 *
	 * @param e extents with which the tensor is reshaped.
	 * @param v value which is appended if the tensor is enlarged.
	 */
	BOOST_UBLAS_INLINE
	void reshape (extents_type const& e, value_type v = value_type{})
	{
		this->extents_ = e;
		this->strides_ = strides_type(this->extents_);

		if(e.product() != this->size())
			this->data_.resize (this->extents_.product(), v);
	}





	friend void swap(tensor& lhs, tensor& rhs) {
		std::swap(lhs.data_   , rhs.data_   );
		std::swap(lhs.extents_, rhs.extents_);
		std::swap(lhs.strides_, rhs.strides_);
	}


	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator begin () const {
		return data_.begin ();
	}

	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cbegin () const {
		return data_.cbegin ();
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator end () const {
		return data_.end();
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cend () const {
		return data_.cend ();
	}

	/// \brief Return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	iterator begin () {
		return data_.begin();
	}

	/// \brief Return an iterator at the end of the tensor
	BOOST_UBLAS_INLINE
	iterator end () {
		return data_.end();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rbegin () const {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crbegin () const {
		return data_.crbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rend () const {
		return data_.rend();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crend () const {
		return data_.crend();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rbegin () {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rend () {
		return data_.rend();
	}


#if 0
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

	extents_type extents_;
	strides_type strides_;
	array_type data_;
};




}}} // namespaces





#endif
