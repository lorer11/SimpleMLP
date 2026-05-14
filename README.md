# SimpleMLP

a minimalistic implementation of a multilayer perceptron (MLP) in C++ without third-party libraries.

### Features
- **backpropagation:** hand-crafted gradient descent implementation.
- **pure C++:** no dependencies.
- **randomization:** weight initialization via `std::mt19937`

### Usage
```bash
g++ main.cpp -o SimpleMLP
./SimpleMLP
