# QuantumGP
Using Quantum Speed-Ups to accelerate Gaussian Process Regression (GPR). The posterior distribution can be obtained 
using Gaussian Inference. This includes the inversion of the covariance matrix. The inversion is accelerated using 
classical and quantum algorithms.

Quantum Solvers
- Harrow-Hassidim-Lloyd (HHL) 
- Variational Quantum Linear Solver (VQLS)

Classical Solvers
- Cholesky
- Partial Cholesky
- Conjugate Gradient (CG)
- Preconditioned Conjugate Gradient (PCG)

Preconditioners
- Partial Cholesky