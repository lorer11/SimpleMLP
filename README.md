# SimpleMLP

a simple multi-layer (MLP) implementation in c++ from scratch.

### Features
- **backpropagation:** manual implementation of gradient descent
- **pure c++:** no external libraries
- **optimization:** uses constant references for performance
- **randomization:** weights initialized via `std::mt19937`

### usage
compile with any c++ compiler:
```bash
g++ main.cpp -o SimpleMLP
./SimpleMLP
