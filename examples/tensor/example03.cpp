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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>

int main()
{
	using namespace boost::numeric::ublas;

	using format  = column_major;
	using tensorf = tensor<float,format>;
	using matrixf = matrix<float,format>;
	using vectorf = vector<float>;

	// Tensor-Vector-Multiplications - Including Transposition
	{

		auto n = shape{3,4,2};
		auto A = tensorf(n,2);
		auto q = 0u; // contraction mode

		// C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);
		q = 1u;
		tensorf C1 = matrixf(n[1],n[2],2) + prod(q,A,vectorf(n[q-1],1));

		// C2(i,k) = A(i,j,k)*T1(j) + 4;
		q = 2u;
		tensorf C2 = prod(q,A,vectorf(n[q-1],1)) + 4;

		// C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);		
		tensorf C3 = prod(1,prod(1,prod(1,A,vectorf(n[0],1)),vectorf(n[1],1)),vectorf(n[2],1));

		// C4(i,j) = A(k,i,j)*T1(k) + 4;
		q = 1u;
		tensorf C4 = prod(q, trans(A,{2,3,1}),vectorf(n[2],1)) + 4;


		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);" << std::endl << std::endl;
		std::cout << "C1=" << C1 << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C2(i,k) = A(i,j,k)*T1(j) + 4;" << std::endl << std::endl;
		std::cout << "C2=" << C2 << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);" << std::endl << std::endl;
		std::cout << "C3()=" << C3(0) << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C4(i,j) = A(k,i,j)*T1(k) + 4;" << std::endl << std::endl;
		std::cout << "C4=" << C4 << ";" << std::endl << std::endl;

	}


	// Tensor-Matrix-Multiplications - Including Transposition
	{

		auto n = shape{3,4,2};
		auto A = tensorf(n,2);
		auto m = 5u;
		auto q = 0u; // contraction mode

		// C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);
		q = 1u;
		tensorf C1 = tensorf(shape{m,n[1],n[2]},2) + prod(q,A,matrixf(m,n[q-1],1));

		// C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;
		q = 2u;
		tensorf C2 = prod(q,A,matrixf(m,n[q-1],1)) + 4;

		// C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);
		q = 3u;
		tensorf C3 = prod(q,prod(q-1,A,matrixf(m+1,n[q-2],1)),matrixf(m+2,n[q-1],1));

		// C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);
		tensorf C4 = prod(q-1,prod(q,A,matrixf(m+2,n[q-1],1)),matrixf(m+1,n[q-2],1));

		// C5(i,k,l) = A(i,k,j)*T1(l,j) + 4;
		q = 3u;
		tensorf C5 = prod(q,trans(A,{1,3,2}),matrixf(m,n[1],1)) + 4;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);" << std::endl << std::endl;
		std::cout << "C1=" << C1 << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;" << std::endl << std::endl;
		std::cout << "C2=" << C2 << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);" << std::endl << std::endl;
		std::cout << "C3=" << C3 << ";" << std::endl << std::endl;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);" << std::endl << std::endl;
		std::cout << "C4=" << C4 << ";" << std::endl << std::endl;
		std::cout << "% C3 and C4 should have the same values, true? " << std::boolalpha << (C3 == C4) << "!" << std::endl;


		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C5(i,k,l) = A(i,k,j)*T1(l,j) + 4;" << std::endl << std::endl;
		std::cout << "C5=" << C5 << ";" << std::endl << std::endl;
	}
}
