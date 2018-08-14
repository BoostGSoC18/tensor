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


#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_TAGS_
#define _BOOST_NUMERIC_UBLAS_TENSOR_TAGS_

#include <stddef.h>
#include "../functional.hpp"

#include <iterator>

namespace boost   {
namespace numeric {
namespace ublas   {
namespace detail  {
namespace tag     {

struct unit_access    {};
struct non_unit_access{};

}
}
}
}
}


//namespace boost   {
//namespace numeric {
//namespace ublas   {

//template <class Z, class D> struct basic_row_major;
//template <class Z, class D> struct basic_column_major;

//}
//}
//}


namespace boost   {
namespace numeric {
namespace ublas   {
namespace tag     {


using first_order = column_major;//basic_column_major<std::size_t,std::ptrdiff_t>;
using last_order = row_major;//basic_row_major<std::size_t,std::ptrdiff_t>;

struct sliced  {};
struct strided {};

}
}
}
}

#endif // FHG_range_H
