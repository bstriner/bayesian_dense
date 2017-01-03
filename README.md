# bayesian_dense
## Bayesian Weight Uncertainty for Keras
The `BayesianDense` layer is a `Dense` layer parameterized by a weight distribution, instead of a point estimate.

Each `BayesianDense` layer learns a Gaussian distribution over weights that can be regularized.

Blundell et. al., Weight Uncertainty in Neural Networks, 
https://arxiv.org/pdf/1505.05424.pdf
