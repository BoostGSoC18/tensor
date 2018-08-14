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


/// \file subtensor.hpp Definition for the tensor template class

#ifndef _BOOST_NUMERIC_UBLAS_SUBTENSOR_HPP_
#define _BOOST_NUMERIC_UBLAS_SUBTENSOR_HPP_




#include <boost/config.hpp>

#include <initializer_list>

#include "algorithms.hpp"
#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "extents.hpp"
#include "strides.hpp"
#include "index.hpp"
#include "span.hpp"



namespace boost { namespace numeric { namespace ublas {

template<class T, class F, class A>
class tensor;

template<class T, class F, class A>
class matrix;

template<class T, class A>
class vector;





/** @brief A view of a dense tensor of values of type \c T.
	*
	* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
	* @tparam F
	* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
*/
template<class S, class T>
class subtensor;


/** @brief A sliced view of a dense tensor of values of type \c T.
		*
		* For a \f$n\f$-dimensional tensor \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
		* to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
		* Elements are constructed by \c A, which need not initialise their value.
		*
		* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
		* @tparam F type of the layout which can be either
		* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
		*/
template<class T, class F, class A>
class subtensor <sliced_tag, tensor<T,F,A>>
		: public detail::tensor_expression<
				subtensor<sliced_tag,tensor<T,F,A>> ,
				subtensor<sliced_tag,tensor<T,F,A>>
				>
{

	static_assert( std::is_same<F,tag::first_order>::value ||
								 std::is_same<F,tag::last_order >::value, "boost::numeric::tensor template class only supports first- or last-order storage formats.");

	using tensor_type = tensor<T,F,A>;
	using self_type  = subtensor<sliced_tag, tensor_type>;
public:

	using domain_tag = sliced_tag;

	using span_type = span<domain_tag,std::size_t>;

	template<class derived_type>
	using tensor_expression_type = detail::tensor_expression<self_type,derived_type>;

	template<class derived_type>
	using matrix_expression_type = matrix_expression<derived_type>;

	template<class derived_type>
	using vector_expression_type = vector_expression<derived_type>;

	using super_type = tensor_expression_type<self_type>;

//	static_assert(std::is_same_v<tensor_expression_type<self_type>, detail::tensor_expression<tensor<T,F,A>,tensor<T,F,A>>>, "tensor_expression_type<self_type>");

	using array_type      = typename tensor_type::array_type;
	using layout_type     = typename tensor_type::layout_type;

	using size_type       = typename tensor_type::size_type;
	using difference_type = typename tensor_type::difference_type;
	using value_type      = typename tensor_type::value_type;

	using reference       = typename tensor_type::reference;
	using const_reference = typename tensor_type::const_reference;

	using pointer         = typename tensor_type::pointer;
	using const_pointer   = typename tensor_type::const_pointer;

//	using iterator        = typename array_type::iterator;
//	using const_iterator  = typename array_type::const_iterator;

//	using reverse_iterator        = typename array_type::reverse_iterator;
//	using const_reverse_iterator  = typename array_type::const_reverse_iterator;

	using tensor_temporary_type = self_type;
	using storage_category = dense_tag;

	using strides_type = basic_strides<std::size_t,layout_type>;
	using extents_type = shape;

	using matrix_type  = matrix<value_type,layout_type,array_type>;
	using vector_type  = vector<value_type,array_type>;



	/** @brief Deleted constructor of a subtensor */
	subtensor () = delete;

	/** @brief Constructs a tensor view from a tensor without any range.
	 *
	 * @note can be regarded as a pointer to a tensor
	 */
	explicit BOOST_UBLAS_INLINE
	subtensor (tensor_type const& t)
		: super_type()
		, tensor_  (t)
		, extents_ (t.extents())
		, strides_ (t.strides())
	{
	}

#if 0

	/** @brief Constructs a tensor view from a tensor without any range.
	 *
	 * @note can be regarded as a pointer to a tensor
	 */
//	explicit BOOST_UBLAS_INLINE
//	subtensor (tensor_type const& t)
//		: tensor_expression_type<self_type>()
//		, tensor_  (t)
//		, extents_ (t.extents())
//		, strides_ (t.strides())
//		, data_    (t.data())
//	{
//	}


	/** @brief Constructs a tensor with a \c shape and initiates it with one-dimensional data
	 *
	 * @code tensor<float> A{extents{4,2,3}, array }; @endcode
	 *
	 *
	 *  @param s initial tensor dimension extents
	 *  @param a container of \c array_type that is copied according to the storage layout
	 */
	BOOST_UBLAS_INLINE
	tensor (extents_type const& s, const array_type &a)
		: tensor_expression_type<self_type>() //tensor_container<self_type>()
		, extents_ (s)
		, strides_ (extents_)
		, data_    (a)
	{
		if(this->extents_.product() != this->data_.size())
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: size of provided data and specified extents do not match.");
	}



	/** @brief Constructs a tensor using a shape tuple and initiates it with a value.
	 *
	 *  @code tensor<float> A{extents{4,2,3}, 1 }; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param i initial value of all elements of type \c value_type
	 */
	BOOST_UBLAS_INLINE
	tensor (extents_type const& e, const value_type &i)
		: tensor_expression_type<self_type>() //tensor_container<self_type> ()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product(), i)
	{}



	/** @brief Constructs a tensor from another tensor
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



	/** @brief Constructs a tensor from another tensor
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


	/** @brief Constructs a tensor with a matrix
	 *
	 * \note Initially the tensor will be two-dimensional.
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
		if(!data_.empty()){
			extents_ = extents_type{v.size1(),v.size2()};
			strides_ = strides_type(extents_);
		}
	}

	/** @brief Constructs a tensor with a matrix
	 *
	 * \note Initially the tensor will be two-dimensional.
	 *
	 *  @param v matrix to be moved.
	 */
	BOOST_UBLAS_INLINE
	tensor (matrix_type &&v)
		: tensor_expression_type<self_type>()
		, extents_ {}
		, strides_ {}
		, data_    {}
	{
		if(v.size1()*v.size2() != 0){
			extents_ = extents_type{v.size1(),v.size2()};
			strides_ = strides_type(extents_);
			data_    = std::move(v.data());
		}
	}

	/** @brief Constructs a tensor using a \c vector
	 *
	 * @note It is assumed that vector is column vector
	 * @note Initially the tensor will be one-dimensional.
	 *
	 *  @param v vector to be copied.
	 */
	BOOST_UBLAS_INLINE
	tensor (const vector_type &v)
		: tensor_expression_type<self_type>()
		, extents_ ()
		, strides_ ()
		, data_    (v.data())
	{
		if(!data_.empty()){
			extents_ = extents_type{data_.size(),1};
			strides_ = strides_type(extents_);
		}
	}

	/** @brief Constructs a tensor using a \c vector
	 *
	 *  @param v vector to be moved.
	 */
	BOOST_UBLAS_INLINE
	tensor (vector_type &&v)
		: tensor_expression_type<self_type>()
		, extents_ {}
		, strides_ {}
		, data_    {}
	{
		if(v.size() != 0){
			extents_ = extents_type{v.size(),1};
			strides_ = strides_type(extents_);
			data_    = std::move(v.data());
		}
	}


	/** @brief Constructs a tensor with another tensor with a different layout
	 *
	 * @param other tensor with a different layout to be copied.
	 */
	BOOST_UBLAS_INLINE
	template<class other_layout>
	tensor (const tensor<value_type, other_layout> &other)
		: tensor_expression_type<self_type> ()
		, extents_ (other.extents())
		, strides_ (other.extents())
		, data_    (other.extents().product())
	{
		copy(this->rank(), this->extents().data(),
				 this->data(), this->strides().data(),
				 other.data(), other.strides().data());
	}

	/** @brief Constructs a tensor with an tensor expression
	 *
	 * @code tensor<float> A = B + 3 * C; @endcode
	 *
	 * @note type must be specified of tensor must be specified.
	 * @note dimension extents are extracted from tensors within the expression.
	 *
	 * @param expr tensor expression
	 */
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

	/** @brief Constructs a tensor with a matrix expression
	 *
	 * @code tensor<float> A = B + 3 * C; @endcode
	 *
	 * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
	 * @note extents are automatically extracted from the temporary matrix
	 *
	 * @param expr matrix expression
	 */
	BOOST_UBLAS_INLINE
	template<class derived_type>
	tensor (const matrix_expression_type<derived_type> &expr)
		: tensor(  matrix_type ( expr )  )
	{
	}

	/** @brief Constructs a tensor with a vector expression
	 *
	 * @code tensor<float> A = b + 3 * b; @endcode
	 *
	 * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
	 * @note extents are automatically extracted from the temporary matrix
	 *
	 * @param expr vector expression
	 */
	BOOST_UBLAS_INLINE
	template<class derived_type>
	tensor (const vector_expression_type<derived_type> &expr)
		: tensor(  vector_type ( expr )  )
	{
	}

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

	tensor& operator=(const_reference v)
	{
		std::fill(this->begin(), this->end(), v);
		return *this;
	}
#endif


	/** @brief Returns true if the tensor is empty (\c size==0) */
	BOOST_UBLAS_INLINE
	bool empty () const {
		return this->size() == size_type(0);
	}


	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size () const {
		return this->extents_.product();
	}

	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size (size_type r) const {
		return this->extents_.at(r);
	}

	/** @brief Returns the number of dimensions/modes of the tensor */
	BOOST_UBLAS_INLINE
	size_type rank () const {
		return this->extents_.size();
	}

	/** @brief Returns the number of dimensions/modes of the tensor */
	BOOST_UBLAS_INLINE
	size_type order () const {
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


#if 0
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
			return this->data_[detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...)];
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
			return this->data_[detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...)];
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
	 *  @code A(i) = a; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	reference operator()(size_type i){
		return this->data_[i];
	}




	/** @brief Generates a tensor index for tensor contraction
	 *
	 *
	 *  @code auto Ai = A(_i,_j,k); @endcode
	 *
	 *  @param i placeholder
	 *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
	 */
	BOOST_UBLAS_INLINE
	template<std::size_t I, class ... index_types>
	decltype(auto) operator() (index::index_type<I> p, index_types ... ps) const
	{
		constexpr auto N = sizeof...(ps)+1;
		if( N != this->rank() )
			throw std::runtime_error("Error in boost::numeric::ublas::operator(): size of provided index_types does not match with the rank.");

		return std::make_pair( std::cref(*this),  std::make_tuple( p, std::forward<index_types>(ps)... ) );
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

#endif





private:





	tensor_type  tensor_;
	std::vector<span_type> spans_;
	extents_type extents_;
	strides_type strides_;
};






/*! @ \brief Computes a new strides for
 *
 * \note the new stride v is computed v[i] = w[i]*s[i]
 *
*/
template <class layout_type, class span_type>
auto compute_strides(strides<layout_type> const& tensor_strides, std::vector<span_type> const& span_vector)
{
	assert(span_vector.size()>1);
	assert(tensor_strides.size()==span_vector.size());

	using strides_type = strides<layout_type>;
	using base_type = typename strides_type::base_type;

	auto range_strides = base_type(span_vector.size());

	std::transform(tensor_strides.begin(), tensor_strides.end(), span_vector.begin(),  range_strides.begin(),
								 [](auto bs, auto const& vr) { return bs * vr.step(); } );

	return strides_type( range_strides );
}

#if 0
/*! \brief modifies base strides for iterating in Y*.
 *
 * \note _view_strides is required for the subiterator of the multi_array_view, when calling
 * begin(), end() and ind2sub() (Y* -> X*) .
 *
*/
template<class T>
fhg::strides compute_view_strides(fhg::shape const& e, fhg::layout const& l)
{
	assert(e.size()>1);
	assert(e.size()==l.size());
	assert(e.valid());
	assert(l.valid());
	return fhg::strides{ e, l };
}



/*! \brief computes the offsets for iterating directly in Y.
 *
 * \note _strides is required for the multi_iterator and iterator of the multi_array_type, when calling
 * dbegin(), dend().
 *
*/
template <class T>
typename multi_array_view<T>::size_type
multi_array_view<T>::compute_offset(fhg::strides const& tensor_strides, std::vector<view_range> const& ranges)
{
	assert(ranges.size()>1);
	assert(tensor_strides.size()==ranges.size());

	return std::inner_product(ranges.begin(), ranges.end(), tensor_strides.begin(), 0ul,
														std::plus<size_type>(), [](const view_range& vr, size_type bs) {return vr.first * bs; } );
}

/*! @brief Computes the extents of multi_array_view from the selecting entities. */

template <class T>
shape
multi_array_view<T>::compute_extents(std::vector<view_range> const& ranges)
{
	assert(ranges.size() > 1);

	std::vector<size_type> extents (ranges.size());
	std::transform(ranges.begin(), ranges.end(), extents.begin(), [](const view_range& vr) { return vr.size; } );

	return shape( extents );
}


template<class T>
void multi_array_view<T>::extract_ranges(std::vector<range> const& ranges)
{
//	if(ranges.size() != this->rank() )
//		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): length of ranges tuple is not equal to this multi_array_view rank.");

//	if(ranges.size() != this->ranges().size() )
//		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): length of this and other ranges tuple unequal.");

//	if(this->base() == nullptr)
//		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): multi_array pointer equals nullptr.");

	for(auto i = 0u; i < ranges.size(); ++i){
		auto const& range = ranges.at(i);

		try { this->extract_range(range.first, range.step, range.last, i, this->base()->extents().at(i)-1); }
		catch (...) { throw; }
	}
}


template<class T>
void
multi_array_view<T>::extract_ranges(difference_type arg_offset)
{
	const difference_type diff = arg_offset - this->base()->offsets().back();

	if(diff < 0)
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): argument is not well defined.");

	const size_type arg = static_cast<size_type>(diff);
	const size_type extent = this->base()->extents().back()-1ul;

	if(arg_offset != fhg::end && arg > extent )
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): index argument for multi_array_type exceeds dimension.");

	_ranges.back() = (arg_offset == fhg::end ? view_range(extent,extent) : view_range(arg,arg));
}

template<class T>
template<class ... Selectors>
void
multi_array_view<T>::extract_ranges ( difference_type arg_offset, Selectors&& ... selectors )
{
	const size_type pos = this->ranges().size() - sizeof...(selectors) - 1ul;
	const difference_type diff = arg_offset - this->base()->offsets().at(pos);

	if(diff < 0)
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): index argument is less than the index offset.");

	const size_type arg = static_cast<size_type>(diff);
	const size_type extent = this->base()->extents().at(pos)-1ul;

	if(arg_offset != fhg::end && arg > extent )
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): index argument for multi_array_type exceeds dimension.");

	_ranges.at(pos) = (arg_offset == fhg::end ? view_range(extent,extent) : view_range(arg,arg));

	extract_ranges(std::forward<Selectors>(selectors) ...);
}

template<class T>
void
multi_array_view<T>::extract_range(difference_type start, difference_type step, difference_type finish, size_type pos, size_type extent)
{
	difference_type offset = this->base()->offsets().at(pos);

	if(start  == fhg::end)
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range not correctly defined. start should not be end.");
	if(finish == fhg::all)
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range (finish == all) not correctly defined. finish should not be all.");
	if(start == fhg::all && finish != fhg::end)
		throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range (finish == end) not correctly defined. if start is all, finish souhld be end.");


	if(start == fhg::all && finish == fhg::end) {
		_ranges.at(pos) = view_range(0ul, extent);
	}
	else if(start != fhg::all && finish == fhg::end) {
		if( start < offset)
			throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range not correctly defined: start should not be less than the index offset.");

		_ranges.at(pos) = view_range(start-offset, step, extent);
	}
	else {
		if( start < offset)
			throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range not correctly defined: start should not be less than the index offset.");
		if( start > finish)
			throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range not correctly defined: start should not be less than finish.");
		if( finish > static_cast<difference_type>(extent)+offset)
			throw std::runtime_error("Error in multi_array_view::extract_ranges(...): range not correctly defined: finish should be less than extent.");


		_ranges.at(pos) = view_range(start-offset, step, finish-offset);
	}
}


template<class T>
void
multi_array_view<T>::extract_ranges(range&& arg)
{
	const size_type extent = _multi_array->extents().back()-1ul;
	const size_type pos    = _multi_array->extents().size()-1ul;
	try { this->extract_range(arg.first, arg.step, arg.last, pos, extent); }
	catch(...) { throw; }
}

template<class T>
template<class ... Selectors>
void
multi_array_view<T>::extract_ranges (range&& arg, Selectors&& ... selectors)
{
	const size_type pos    = _ranges.size() - sizeof...(selectors) - 1ul;
	const size_type extent = _multi_array->extents().at(pos)-1;
	try { this->extract_range(arg.first, arg.step, arg.last, pos, extent); }
	catch(...) { throw; }
	extract_ranges(std::forward<Selectors>(selectors)...);
}
#endif

}}} // namespaces






#endif
