//  Copyright (c) 2018
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which firsted as a Google Summer of Code project.
//


/// \file subtensor.hpp Definition for the tensor template class

#ifndef _BOOST_NUMERIC_UBLAS_SUBTENSOR_HPP_
#define _BOOST_NUMERIC_UBLAS_SUBTENSOR_HPP_




#include <boost/config.hpp>

#include <initializer_list>

#include "algorithms.hpp"
#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "subtensor_utility.hpp"
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
class subtensor <tag::sliced, tensor<T,F,A>>
		: public detail::tensor_expression<
				subtensor<tag::sliced,tensor<T,F,A>> ,
				subtensor<tag::sliced,tensor<T,F,A>>
				>
{

	static_assert( std::is_same<F,tag::first_order>::value ||
								 std::is_same<F,tag::last_order >::value, "boost::numeric::tensor template class only supports first- or last-order storage formats.");

	using tensor_type = tensor<T,F,A>;
	using self_type  = subtensor<tag::sliced, tensor_type>;
public:

	using domain_tag = tag::sliced;

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
	 */
	BOOST_UBLAS_INLINE
	subtensor (tensor_type& t)
		: super_type    ()
		, spans_        ()
		, extents_      (t.extents())
		, strides_      (t.strides())
		, span_strides_ (t.strides())
		, data_         (t.data())
	{
	}

	template<typename ... span_types>
	subtensor(tensor_type& t, span_types&& ... spans)
			: super_type     ()
			, spans_         (generate_span_vector(t.extents(),std::forward<span_types>(spans)...))
			, extents_       (spans_)
			, strides_       (extents_)
			, span_strides_  (strides_,spans_)
	{
//		if( m == nullptr)
//			throw std::length_error("Error in tensor_view<T>::tensor_view : multi_array_type is nullptr.");
//		if( t == nullptr)
//			throw std::length_error("Error in tensor_view<T>::tensor_view : tensor_type is nullptr.");
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
		return this->data_;
	}

	/** @brief Returns a \c const reference to the container. */
	BOOST_UBLAS_INLINE
	pointer data () {
		return this->data_;
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

	std::vector<span_type> spans_;
	extents_type extents_;
	strides_type strides_;
	strides_type span_strides_;
	pointer data_;
};





/*! @brief Computes span strides for a subtensor
 *
 * span stride v is computed according to: v[i] = w[i]*s[i], where
 * w[i] is the i-th stride of the tensor
 * s[i] is the step size of the i-th span
 *
 * @param[in] strides strides of the tensor this subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<class strides_type, class span_type>
auto span_strides(strides_type const& strides, std::vector<span_type> const& spans)
{
	if(strides.size() != spans.size())
		throw std::runtime_error("Error in boost::numeric::ublas::subtensor::span_strides(): tensor strides.size() != spans.size()");

	using base_type = typename strides_type::base_type;
	auto span_strides = base_type(spans.size());

	std::transform(strides.begin(), strides.end(), spans.begin(), span_strides.begin(),
								 [](auto w, auto const& s) { return w * s.step(); } );

	return strides_type( span_strides );
}

/*! @brief Computes the data pointer offset for a subtensor
 *
 * offset is computed according to: sum ( f[i]*w[i] ), where
 * f[i] is the first element of the i-th span
 * w[i] is the i-th stride of the tensor
 *
 * @param[in] strides strides of the tensor this subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<class strides_type, class span_type>
auto offset(strides_type const& strides, std::vector<span_type> const& spans)
{
	if(strides.size() != spans.size())
		throw std::runtime_error("Error in boost::numeric::ublas::subtensor::offset(): tensor strides.size() != spans.size()");

	using value_type = typename strides_type::value_type;

	return std::inner_product(spans.begin(), spans.end(), strides.begin(), value_type(0),
														std::plus<value_type>(), [](span_type const& s, value_type w) {return s.first() * w; } );
}


/*! @brief Computes the extents of the subtensor.
 *
 * i-th extent is given by span[i].size()
 *
 * @param[in] spans vector of spans of the subtensor
 */
template<class span_type>
auto extents(std::vector<span_type> const& spans)
{
	using base_type  = typename shape::base_type;
	auto extents = base_type(spans.size());
	std::transform(spans.begin(), spans.end(), extents.begin(), [](span_type const& s) { return s.size(); } );
	return shape( extents );
}


/*! @brief Auxiliary function for subtensor which possibly transforms a span instance
 *
 * transform_span(span()     ,4) -> span(0,3)
 * transform_span(span(1,1)  ,4) -> span(1,1)
 * transform_span(span(1,3)  ,4) -> span(1,3)
 * transform_span(span(2,end),4) -> span(2,3)
 * transform_span(span(end)  ,4) -> span(3,3)
 *
 * @note span is zero-based indexed.
 *
 * @param[in] s      span that is going to be transformed
 * @param[in] extent extent that is maybe used for the tranformation
 */
template<class size_type, class span_tag>
auto transform_span(span<span_tag, size_type> const& s, size_type const extent)
{
	using span_type = span<span_tag, size_type>;

	size_type first = s.first();
	size_type last  = s.last ();
	size_type size  = s.size ();

	auto const extent0 = extent-1;


	if constexpr (std::is_same<span_tag,tag::sliced>::value ){
		if(size == 0)        return span_type(0       , extent0);
		else if(first== end) return span_type(extent0 , extent0);
		else if(last == end) return span_type(first   , extent0);
		else                 return span_type(first   , last  );
	}
	else {
		size_type step  = s.step ();
		if(size == 0)        return span_type(0       , size_type(1), extent0);
		else if(first== end) return span_type(extent0 , step, extent0);
		else if(last == end) return span_type(first   , step, extent0);
		else                 return span_type(first   , step, last  );
	}
}



template<std::size_t r,class extents_type, class span_tag, class size_type, class ... span_types>
void transform_spans_impl(extents_type const& extents,
													std::vector<span<span_tag, size_type>>& span_vector,
													span<span_tag, size_type> const& s,
													span_types&& ... spans)
{
	span_vector.at(r) = transform_span(s,extents.at(r));
	if constexpr (sizeof...(spans)>0)
		transform_spans_impl<r+1>(extents, span_vector, std::forward<span_types>(spans)...);
}

template<std::size_t r, class extents_type, class size_type, class span_type, class ... span_types>
void transform_spans_impl (extents_type const& extents,
													 std::vector<span_type>& span_vector,
													 size_type arg,
													 span_types&& ... spans )
{
	span_vector.at(r) = transform_span(span_type(arg),extents.at(r));
	if constexpr (sizeof...(spans)>0)
		transform_spans_impl<r+1>(extents, span_vector, std::forward<span_types>(spans) ... );
}


/*! @brief Auxiliary function for subtensor that generates vector of spans
 *
 * generate_span_vector<span>(shape(4,3,5,2), span(), 1, span(2,end), end  )
 * -> vector (span(0,3), span(1,1), span(2,4),span(1,1))
 *
 * @note span is zero-based indexed.
 *
 * @param[in] extents of the tensor
 * @param[in] spans spans with which the subtensor is created
 */
template<class span_type, class extents_type, class ... span_types>
auto generate_span_vector(extents_type const& extents, span_types&& ... spans)
{
	constexpr auto n = sizeof...(spans);
	if(extents.size() != n)
		throw std::runtime_error("Error in boost::numeric::ublas::generate_span_vector() when creating subtensor: the number of spans does not match with the tensor rank.");
	std::vector<span_type> span_vector(n);
	if constexpr (n>0)
		transform_spans_impl<n-n>(  extents, span_vector, std::forward<span_types>(spans)... );
	return span_vector;
}



}
}
} // namespaces






#endif
