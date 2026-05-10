# SimpleMLP

A simple Multi-Layer Perceptron (MLP) implementation in C++ from scratch.

### Features
- **Backpropagation:** Manual implementation of gradient descent.
- **Pure C++:** No external libraries, uses only standard headers.
- **Optimization:** Uses constant references for performance.
- **Randomization:** Weights initialized via `std::mt19937`.

### Usage
Compile with any C++11 compiler:
```bash
g++ main.cpp -o SimpleMLP
./SimpleMLP
