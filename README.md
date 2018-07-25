Boost.uBLAS Linear Algebra Library
=====
Boost.uBLAS is part of the [Boost C++ Libraries](http://github.com/boostorg). It is directed towards scientific computing on the level of basic linear algebra constructions with matrices and vectors and their corresponding abstract operations. 

## Fork
This fork contains an additional branch tensor that has been created for adding tensor support as a Google Summer of Code 2018 project. All details about the project and code is found on the [wiki page](https://github.com/BoostGSoC18/tensor/wiki).

## License

Distributed under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).

## Properties

* Header-only
* Requires C++17 compatible compiler
* Unit-tests require Boost.Test

## Build Status

Branch          | Travis | Appveyor | codecov.io | Docs |
:-------------: | ------ | -------- | ---------- | ---- |
[`develop`](https://travis-ci.org/boostorg/ublas.svg?branch=develop) | [![Build Status](https://travis-ci.org/boostorg/ublas.svg?branch=develop)](https://travis-ci.org/boostorg/ublas) | [![Build status](https://ci.appveyor.com/api/projects/status/ctu3wnfowa627ful/branch/develop?svg=true)](https://ci.appveyor.com/project/stefanseefeld/ublas/branch/develop) | [![codecov](https://codecov.io/gh/boostorg/uuid/branch/develop/graph/badge.svg)](https://codecov.io/gh/boostorg/uuid/branch/develop) | [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](http://www.boost.org/doc/libs/release/libs/numeric)
[`tensor`](https://travis-ci.org/BoostGSoC18/tensor.svg?branch=tensor) | [![Build Status](https://travis-ci.org/BoostGSoC18/tensor.svg?branch=tensor)](https://travis-ci.org/BoostGSoC18/tensor) | [![Build status](https://ci.appveyor.com/api/projects/status/github/BoostGSoC18/tensor?svg=true)](https://ci.appveyor.com/api/projects/status/github/BoostGSoC18/tensor) | [![codecov](https://codecov.io/gh/boostorg/uuid/branch/develop/graph/badge.svg)](https://codecov.io/gh/boostorg/uuid/branch/develop) | [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](http://www.boost.org/doc/libs/release/libs/numeric)

## Directories

| Name        | Purpose                        |
| ----------- | ------------------------------ |
| `doc`       | documentation                  |
| `examples`  | example files                  |
| `include`   | headers                        |
| `test`      | unit tests                     |

## More information

* Ask questions in [stackoverflow](http://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-ublas) with `boost-ublas` or `ublas` tags.
* Report [bugs](https://github.com/boostorg/ublas/issues) and be sure to mention Boost version, platform and compiler you're using. A small compilable code sample to reproduce the problem is always good as well.
* Submit your patches as pull requests against **develop** branch. Note that by submitting patches you agree to license your modifications under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).
* Developer discussions about the library are held on the [Boost developers mailing list](https://lists.boost.org/mailman/listinfo.cgi/ublas). Be sure to read the [discussion policy](http://www.boost.org/community/policy.html) before posting and add the `[ublas]` tag at the beginning of the subject line
* For any other questions, you can contact David, Stefan or Cem: david.bellot-AT-gmail-DOT-com, cem.bassoy-AT-gmail-DOT-com stefan-AT-seefeld-DOT-name
