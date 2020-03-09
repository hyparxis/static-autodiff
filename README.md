# adiff

Adiff is a header only C++ library that implements reverse mode automatic differentiation with compile-time static graphs. Because the computation graph is defined at compile time, adiff has extremely good cache performance and incurs no run-time costs like dynamic memory allocation, meaning it can get up to 1000x performance improvements over general libraries like Pytorch C++ in some cases.
