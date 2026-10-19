# Federated Learning Framework

Privacy-preserving Federated Learning with production-grade security.

## Features
- FedAvg: Federated Averaging algorithm
- FedProx: proximal term for heterogeneous data
- Differential Privacy: Gaussian mechanism with (epsilon, delta) budget
- Secure Aggregation: cryptographic masking
- Byzantine Fault Tolerance: Krum aggregation

## Privacy Guarantees
| Mechanism | epsilon | delta | Utility |
|-----------|---------|-------|---------|
| Gaussian | 1.0 | 1e-5 | High |
| Laplace | 0.5 | 0 | Medium |
