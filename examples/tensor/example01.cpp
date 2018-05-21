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

#include <boost/numeric/ublas/tensor.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <ostream>

int main()
{
	using namespace boost::numeric::ublas;
	using namespace boost::multiprecision;


	// creates a three-dimensional tensor with extents 3,4 and 2
	// tensor A stores single-precision floating-point number according
	// to the first-order storage format
	using ftype = float;
	auto A = tensor<ftype>{3,4,2};

	// initializes the tensor with increasing values along the first-index
	// using a single index.
	auto vf = ftype(0);
	for(auto i = 0u; i < A.size(); ++i, vf += ftype(1))
		A[i] = vf;

	// formatted output
	std::cout << "A=" << A << ";" << std::endl;

	// creates a four-dimensional tensor with extents 5,4,3 and 2
	// tensor A stores complex floating-point extended double precision numbers
	// according to the last-order storage format
	using ctype = std::complex<cpp_bin_float_double_extended>;
	auto B = tensor<ctype,last_order>(shape{5,4,3,2},ctype{});

	// initializes the tensor with increasing values along the last-index
	auto vc = ctype(0,0);
	for(auto i = 0u; i < B.size(); ++i, vc += ctype(1,1))
		B[i] = vc;

	// formatted output
	std::cout << "B=" << B << ";" << std::endl;
}
