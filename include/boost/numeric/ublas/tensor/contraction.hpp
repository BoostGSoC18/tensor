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


#ifndef _BOOST_UBLAS_TENSOR_CONTRACTION_
#define _BOOST_UBLAS_TENSOR_CONTRACTION_


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


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


/** @brief Computes the tensor-times-vector product for the contraction mode m > 0
 *
 * Implements C[i1,i2,...,im-1,im+1,...,ip] = sum(A[i1,i2,...,im,...,ip] * b[im])
 *
 * @note is used in function tensor_times_vector
 *
 *
 *
 * @param m  zero-based contraction mode with 0<m<p
 * @param r  zero-based recursion level starting with r=p-1 for tensor A
 * @param q  zero-based recursion level starting with q=p-1 for tensor C
 * @param c  pointer to the output tensor
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
*/

template <class pointer_t_c, class pointer_t_a, class pointer_t_b, class size_t>
void ttv( size_t const m, size_t const r, size_t const q,
					pointer_t_c c, size_t const*const nc, size_t const*const wc,
					pointer_t_a a, size_t const*const na, size_t const*const wa,
					pointer_t_b b)
{

		if(r == m) {
				ttv(m, r-1, q, c, nc, wc,    a, na, wa,    b);
		}
		else if(r == 0){
				for(size_t i0 = 0u; i0 < na[0]; c += wc[0], a += wa[0], ++i0) {
						auto c1 = c; auto a1 = a; auto b1 = b;
						for(size_t im = 0u; im < na[m]; a1 += wa[m], ++b1, ++im)
								*c1 += *a1 * *b1;
				}
		}
		else{
				for(size_t i = 0u; i < na[r]; c += wc[q], a += wa[r], ++i)
						ttv(m, r-1, q-1, c, nc, wc,    a, na, wa,    b);
		}
}


/** @brief Computes the tensor-times-vector product for the contraction mode m = 0
 *
 * Implements C[i2,...,ip] = sum(A[i1,...,ip] * b[i1])
 *
 * @note is used in function tensor_times_vector
 *
 * @param m  zero-based contraction mode with m=0
 * @param r  zero-based recursion level starting with r=p-1
 * @param c  pointer to the output tensor
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
*/
template <class pointer_t_c, class pointer_t_a, class pointer_t_b, class size_t>
void ttv0(size_t const r,
					pointer_t_c c, size_t const*const nc, size_t const*const wc,
					pointer_t_a a, size_t const*const na, size_t const*const wa,
					pointer_t_b b)
{

		if(r > 1){
				for(size_t i = 0u; i < na[r]; c += wc[r-1], a += wa[r], ++i)
						ttv0(r-1, c, nc, wc,    a, na, wa,    b);
		}
		else{
				for(size_t i1 = 0u; i1 < na[1]; c += wc[0], a += wa[1], ++i1)
				{
						auto c1 = c; auto a1 = a; auto b1 = b;
						for(size_t i0 = 0u; i0 < na[0]; a1 += wa[0], ++b1, ++i0)
								*c1 += *a1 * *b1;
				}
		}
}


/** @brief Computes the matrix-times-vector product
 *
 * Implements C[i1] = sum(A[i1,i2] * b[i2]) or C[i2] = sum(A[i1,i2] * b[i1])
 *
 * @note is used in function tensor_times_vector
 *
 * @param m  zero-based contraction mode with m=0 or m=1
 * @param c  pointer to the output tensor
 * @param nc pointer to the extents of tensor c
 * @param wc pointer to the strides of tensor c
 * @param a  pointer to the first input tensor
 * @param na pointer to the extents of input tensor a
 * @param wa pointer to the strides of input tensor a
 * @param b  pointer to the second input tensor
*/
template <class pointer_t_c, class pointer_t_a, class pointer_t_b, class size_t>
void mtv(size_t const m,
				 pointer_t_c c, size_t const*const   , size_t const*const wc,
				 pointer_t_a a, size_t const*const na, size_t const*const wa,
				 pointer_t_b b)
{
		// decides whether matrix multiplied with vector or vector multiplied with matrix
		const auto o = (m == 0) ? 1 : 0;

		for(size_t io = 0u; io < na[o]; c += wc[o], a += wa[o], ++io) {
				auto c1 = c; auto a1 = a; auto b1 = b;
				for(size_t im = 0u; im < na[m]; a1 += wa[m], ++b1, ++im)
						*c1 += *a1 * *b1;
		}
}




template <class pointer_t_a, class pointer_t_b, class value_t>
value_t inner(size_t const r, size_t const*const n,
					 pointer_t_a  a, size_t const*const wa,
					 pointer_t_b  b, size_t const*const wb,
					 value_t sum)
{
		if(r == 0)
				for(size_t i0 = 0u; i0 < n[0]; a += wa[0], b += wb[0], ++i0)
						sum += *a * *b;
		else
				for(size_t ir = 0u; ir < n[r]; a += wa[r], b += wb[r], ++ir)
						 sum = inner(r-1, n,   a, wa,    b, wb, sum);
		return sum;
}


//template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
//void
//outer_2x2(
//				size_t const pa,
//				size_t const rc, pointer_t_c c, const size_t*   , const size_t* wc,
//				size_t const ra, pointer_t_a a, const size_t* na, const size_t* wa,
//				size_t const rb, pointer_t_b b, const size_t* nb, const size_t* wb)
//{
//		assert(rc == 3);
//		assert(ra == 1);
//		assert(rb == 1);

//	for(size_t ib1 = 0u; ib1 < nb[1]; b += wb[1], c += wc[pa+1], ++ib1)
//		{
//				auto c2 = c;
//				auto b0 = b;
//		for(size_t ib0 = 0u; ib0 < nb[0]; b0 += wb[0], c2 += wc[pa], ++ib0)
//				{
//						const auto b = *b0;
//						auto c1 = c2;
//						auto a1 = a;
//						for(size_t ia1 = 0u; ia1 < na[1]; a1 += wa[1], c1 += wc[1], ++ia1)
//						{
//								auto a0 = a1;
//								auto c0 = c1;
//								for(size_t ia0 = 0u; ia0 < na[0]; a0 += wa[0], c0 += wc[0], ++ia0){
//										*c0 = *a0 * b;
//								}
//						}
//				}
//		}
//}

//template<class pointer_t_c, class pointer_t_a, class pointer_t_b>
//void
//outer_recursion(
//				size_t const pa,
//				size_t const rc, pointer_t_c c, const size_t* nc, const size_t* wc,
//				size_t const ra, pointer_t_a a, const size_t* na, const size_t* wa,
//				size_t const rb, pointer_t_b b, const size_t* nb, const size_t* wb)
//{
//		if(rb > 1) // ra > 1 &&
//		{
//				for(size_t ib = 0u; ib < nb[rb]; b += wb[rb], c += wc[rc], ++ib)
//			 outer_recursion(pa, rc-1, c, nc, wc,    ra, a, na, wa,    rb-1, b, nb, wb);
//		}
//		else if(ra > 1) //  && rb == 1
//		{
//				for(size_t ia = 0u; ia < na[ra]; a += wa[ra], c += wc[ra], ++ia)
//			 outer_recursion(pa, rc-1, c, nc, wc,   ra-1, a, na, wa,   rb, b, nb, wb);
//		}
//		else
//		{
//				assert(ra == 1 && rb == 1 && rc == 3);
//		outer_2x2(pa, rc, c, nc, wc,   ra, a, na, wa,    rb, b, nb, wb);
//		}
//}

} // namespace detail
} // namespace ublas
} // namespace numeric
} // namespace boost




//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


#include <stdexcept>

namespace boost {
namespace numeric {
namespace ublas {

/** @brief Computes the tensor-times-vector product
 *
 * Implements
 *   C[i1,i2,...,im-1,im+1,...,ip] = sum(A[i1,i2,...,im,...,ip] * b[im]) with m>1 and
 *   C[i2,...,ip]                  = sum(A[i1,...,ip]           * b[i1]) with m=1
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
void ttv(std::size_t const m, std::size_t const p,
				 pointer_t_c c,       std::size_t const*const nc, std::size_t const*const wc,
				 const pointer_t_a a, std::size_t const*const na, std::size_t const*const wa,
				 const pointer_t_b b, std::size_t const*const nb, std::size_t const*const /*wb*/)
{
	static_assert( std::is_pointer<pointer_t_c>::value & std::is_pointer<pointer_t_a>::value & std::is_pointer<pointer_t_b>::value,
										 "Static error in boost::numeric::ublas::tensor_times_vector: Argument types for pointers are not pointer types.");

	if( m == 0)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Contraction mode must be greater than zero.");

	if( p < m )
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Rank must be greater equal the modus.");

	if( p == 0)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Rank must be greater than zero.");

	if(c == nullptr || a == nullptr || b == nullptr)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Pointers shall not be null pointers.");

	for(size_t i = 0; i < m-1; ++i)
		if(na[i] != nc[i])
			throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extents (except of dimension mode) of A and C must be equal.");

	for(size_t i = m; i < p; ++i)
		if(na[i] != nc[i-1])
			throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extents (except of dimension mode) of A and C must be equal.");

	const auto max = std::max(nb[0], nb[1]);
	if(  na[m-1] != max)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extent of dimension mode of A and b must be equal.");


	if((m != 1) && (p > 2))
		detail::ttv(m-1, p-1, p-2, c, nc, wc,    a, na, wa,   b);
	else if ((m == 1) && (p > 2))
		detail::ttv0(p-1, c, nc, wc,  a, na, wa,   b);
	else
		detail::mtv(m-1, c, nc, wc,  a, na, wa,   b);

}



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

template <class pointer_t_a, class pointer_t_b, class pointer_t_c>
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



template <class pointer_t_a, class pointer_t_b, class value_t>
auto inner(size_t const p, size_t const*const n,
					 pointer_t_a  a, size_t const*const wa,
					 pointer_t_b  b, size_t const*const wb,
					 value_t sum)
{
	static_assert( std::is_pointer<pointer_t_a>::value && std::is_pointer<pointer_t_b>::value,
								 "Static error in boost::numeric::ublas::inner: argument types for pointers must be pointer types.");

		if(p<2)
				throw std::length_error("Error in boost::numeric::ublas::inner: Rank must be greater than zero.");

		if(a == nullptr || b == nullptr)
				throw std::length_error("Error in boost::numeric::ublas::inner: Pointers shall not be null pointers.");

		return detail::inner(p-1, n, a, wa, b, wb, sum);

}


}
}
}

#endif
