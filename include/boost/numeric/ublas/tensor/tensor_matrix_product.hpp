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


#ifndef _BOOST_UBLAS_TENSOR_MATRIX_PRODUCT_
#define _BOOST_UBLAS_TENSOR_MATRIX_PRODUCT_


namespace boost {
namespace numeric {
namespace ublas {
namespace detail {




/** @brief Computes the tensor-times-matrix product for the contraction mode m > 0
 *
 * Implements C[i1,i2,...,im-1,j,im+1,...,ip] = sum(A[i1,i2,...,im,...,ip] * B[j,im])
 *
 * @note is used in function tensor_times_matrix
 *
 * @param m  zero-based contraction mode with 0<m<p
 * @param r  zero-based recursion level starting with r=p-1
 * @param c  pointer to the output tensor
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
 * @param nb pointer to the extents of input tensor b
 * @param wb pointer to the strides of input tensor b
*/

template <class pointer_t_c, class pointer_t_a, class pointer_t_b, class size_t>
void ttm(size_t const m,  size_t const r,
				 pointer_t_c c, size_t const*const nc, size_t const*const wc,
				 pointer_t_a a, size_t const*const na, size_t const*const wa,
				 pointer_t_b b, size_t const*const nb, size_t const*const wb)
{

		if(r == m) {
				ttm(m, r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
		}


		else if(r == 0){
				for(size_t i0 = 0u; i0 < nc[0]; c += wc[0], a += wa[0], ++i0) {

						auto cm = c;
						auto b0 = b;

						// r == m
						for(size_t i0 = 0u; i0 < nc[m]; cm += wc[m], b0 += wb[0], ++i0){

								auto am = a;
								auto b1 = b0;

								for(size_t i1 = 0u; i1 < nb[1]; am += wa[m], b1 += wb[1], ++i1){
										*cm += *am * *b1;
								}
						}
				}
		}

		else{
				for(size_t i = 0u; i < na[r]; c += wc[r], a += wa[r], ++i)
						ttm(m, r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
		}
}

/** @brief Computes the tensor-times-matrix product for the contraction mode m = 0
 *
 * Implements C[j,i2,...,ip] = sum(A[i1,i2,...,ip] * B[j,i1])
 *
 * @note is used in function tensor_times_matrix
 *
 * @param m  zero-based contraction mode with 0<m<p
 * @param r  zero-based recursion level starting with r=p-1
 * @param c  pointer to the output tensor
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
 * @param nb pointer to the extents of input tensor b
 * @param wb pointer to the strides of input tensor b
*/
template <class pointer_t_c, class pointer_t_a, class pointer_t_b, class size_t>
void ttm0( size_t const r,
					 pointer_t_c c, size_t const*const nc, size_t const*const wc,
					 pointer_t_a a, size_t const*const na, size_t const*const wa,
					 pointer_t_b b, size_t const*const nb, size_t const*const wb)
{

		if(r > 1){
				for(size_t i = 0u; i < na[r]; c += wc[r], a += wa[r], ++i)
						ttm0(r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
		}
		else{
				for(size_t i1 = 0u; i1 < nc[1]; c += wc[1], a += wa[1], ++i1) {
						auto cm = c;
						auto b0 = b;
						// r == m == 0
						for(size_t i0 = 0u; i0 < nc[0]; cm += wc[0], b0 += wb[0], ++i0){

								auto am = a;
								auto b1 = b0;
								for(size_t i1 = 0u; i1 < nb[1]; am += wa[0], b1 += wb[1], ++i1){

										*cm += *am * *b1;
								}
						}
				}
		}
}


}
}
}
}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>
#include <vector>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>



namespace boost {
namespace numeric {
namespace ublas {


/** @brief Computes the tensor-times-matrix product
 *
 * Implements
 *   C[i1,i2,...,im-1,j,im+1,...,ip] = sum(A[i1,i2,...,im,...,ip] * B[j,im]) with m>1 and
 *   C[j,i2,...,ip]                  = sum(A[i1,i2,...,ip]        * B[j,i1]) with m=1
 *
 *
 * @param m  contraction mode with 0 < m <= p
 * @param p  number of dimensions (rank) of the first input tensor with p > 0
 * @param c  pointer to the output tensor with rank p-1
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
 * @param nb pointer to the extents of input tensor b
 * @param wb pointer to the strides of input tensor b
*/

template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void ttm(size_t const m, size_t const p,
				 pointer_t_c c,       size_t const*const nc, size_t const*const wc,
				 const pointer_t_a a, size_t const*const na, size_t const*const wa,
				 const pointer_t_b b, size_t const*const nb, size_t const*const wb)
{

		static_assert( std::is_pointer<pointer_t_c>::value & std::is_pointer<pointer_t_a>::value & std::is_pointer<pointer_t_b>::value,
											 "Static error in boost::numeric::ublas::tensor_times_matrix: Argument types for pointers are not pointer types.");

		if( m == 0 )
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: Contraction mode must be greater than zero.");

		if( p < m )
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: Rank must be greater equal than the specified mode.");

		if( p == 0)
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix:Rank must be greater than zero.");

		if(c == nullptr || a == nullptr || b == nullptr)
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: Pointers shall not be null pointers.");

		for(size_t i = 0; i < m-1; ++i)
				if(na[i] != nc[i])
						throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: Extents (except of dimension mode) of A and C must be equal.");

		for(size_t i = m; i < p; ++i)
				if(na[i] != nc[i])
						throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: Extents (except of dimension mode) of A and C must be equal.");

		if(na[m-1] != nb[1])
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: 2nd Extent of B and M-th Extent of A must be the equal.");

		if(nc[m-1] != nb[0])
				throw std::length_error("Error in boost::numeric::ublas::tensor_times_matrix: 1nd Extent of B and M-th Extent of C must be the equal.");

		if(m != 1)
				detail::ttm(m-1, p-1, c, nc, wc,    a, na, wa,   b, nb, wb);
		else
				detail::ttm0(p-1, c, nc, wc,    a, na, wa,   b, nb, wb);

}



template<class Value, class Format, class Allocator>
class tensor;

template<class Value, class Format, class Allocator>
class matrix;


template<class V, class F, class A>
tensor<V,F,A> prod(const std::size_t m, tensor<V,F,A> const& a, matrix<V,F,A> const& b)
{

	auto const p = a.rank();

	if( m == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Contraction mode must be greater than zero.");

	if( p < m )
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater equal the modus.");

	if( p == 0)
		throw std::length_error("Error in boost::numeric::ublas::prod: Rank must be greater than zero.");

	auto nc = a.extents().base();
	auto nb = shape      {b.size1(),b.size2()};
	auto wb = strides<F> (nb);

	nc[m-1] = nb[0];

	auto c = tensor<V,F,A>(shape(nc),V{});

	ttm(m, p,
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			b.data().data(), nb.data(), wb.data());


	return c;
}


}
}
}

#endif
