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

#include <iostream>

namespace boost {
namespace numeric {
namespace ublas {
namespace detail {

template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void ttv_recursion_notzero(
        const size_t m,  size_t r, size_t q,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b)
{

    if(r == m) {
        ttv_recursion_notzero(m, r-1, q, c, nc, wc,    a, na, wa,    b);
    }
    else if(r == 0){
        for(size_t i0 = 0u; i0 < na[0]; c += wc[0], a += wa[0], ++i0) {
            auto c1 = c;
            auto a1 = a;
            auto b1 = b;
            for(size_t im = 0u; im < na[m]; a1 += wa[m], ++b1, ++im)
                *c1 += *a1 * *b1;
        }
    }
    else{
        for(size_t i = 0u; i < na[r]; c += wc[q], a += wa[r], ++i)
            ttv_recursion_notzero(m, r-1, q-1, c, nc, wc,    a, na, wa,    b);
    }
}

template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void ttv_recursion_zero(
        size_t r,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b)
{

    if(r > 1){
        for(size_t i = 0u; i < na[r]; c += wc[r-1], a += wa[r], ++i)
            ttv_recursion_zero(r-1, c, nc, wc,    a, na, wa,    b);
    }
    else{
        for(size_t i1 = 0u; i1 < na[1]; c += wc[0], a += wa[1], ++i1)
        {
            auto c1 = c;
            auto a1 = a;
            auto b1 = b;
            for(size_t i0 = 0u; i0 < na[0]; a1 += wa[0], ++b1, ++i0)
                *c1 += *a1 * *b1;
        }
    }
}

template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void mtv(const size_t m,
         pointer_t_c c, const size_t*   , const size_t* wc,
         pointer_t_a a, const size_t* na, const size_t* wa,
         pointer_t_b b)
{
    assert(m == 1 || m == 0);

    // decides whether matrix multiplied with vector or vector multiplied with matrix
    const auto o = (m == 0) ? 1 : 0;

    for(size_t io = 0u; io < na[o]; c += wc[o], a += wa[o], ++io) {
        auto c1 = c;
        auto a1 = a;
        auto b1 = b;
        // r == m
        for(size_t im = 0u; im < na[m]; a1 += wa[m], ++b1, ++im)
            *c1 += *a1 * *b1;
    }
}

} // namespace detail
} // namespace ublas
} // namespace numeric
} // namespace boost


namespace boost {
namespace numeric {
namespace ublas {



template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
tensor_times_vector(size_t m, size_t p,
					pointer_t_c c, const size_t* nc, const size_t* wc,
					pointer_t_a a, const size_t* na, const size_t* wa,
					pointer_t_b b, const size_t* nb, const size_t* /*wb*/)
{

	if( p < m )
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Rank must be greater equal the modus.");

	if( p == 0)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Rank must be greater than zero.");

	if(c == nullptr || a == nullptr || b == nullptr)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Pointers shall not be null pointers.");

	for(size_t i = 0; i < m-1; ++i){
		if(na[i] != nc[i])
			throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extents (except of dimension mode) of A and C must be equal.");
	}

	for(size_t i = m; i < p; ++i){
		if(na[i] != nc[i-1])
			throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extents (except of dimension mode) of A and C must be equal.");
	}

	const auto max = std::max(nb[0], nb[1]);
	if(  na[m-1] != max)
		throw std::length_error("Error in boost::numeric::ublas::tensor_times_vector: Extent of dimension mode of A and b must be equal.");


	if((m != 1) && (p > 2))
		detail::ttv_recursion_notzero(m-1, p-1, p-2, c, nc, wc,    a, na, wa,   b);
	else if ((m == 1) && (p > 2))
		detail::ttv_recursion_zero(p-1, c, nc, wc,    a, na, wa,   b);
	else
		detail::mtv(m-1, c, nc, wc,    a, na, wa,   b);

}



}
}
}


#endif
