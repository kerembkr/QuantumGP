# QuantumGP
Using Quantum Speed-Ups to accelerate Gaussian Process Regression (GPR). The posterior distribution can be obtained 
using Gaussian Inference. This includes the inversion of the covariance matrix. The inversion is accelerated using 
classical and quantum algorithms.

Quantum Solvers
- Variational Quantum Linear Solver (VQLS)
- Fast & Slow VQLS (using global and local optimization techniques in a hybrid fashion)
- Basin Hopping VQLS (using random perturbations to escape local minima)
- Deep VQLS (using deep neural networks to initialize the weights)

Classical Solvers
- Cholesky
- Partial Cholesky
- Conjugate Gradient (CG)
- Preconditioned Conjugate Gradient (PCG)

Preconditioners
- Partial Cholesky