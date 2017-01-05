# bayesian_dense
## Bayesian Weight Uncertainty for Keras

The `BayesianDense` layer is a `Dense` layer parameterized by a weight distribution, instead of a point estimate. Each `BayesianDense` layer learns a Gaussian distribution over weights and biases that can be regularized.

`VariationalRegularizer` is an exemplary regularizer calculating `-0.5 * mean(1 + p - K.exp(p))` where `p` is log of sigma squared. This is just a simple regularizer. Please experiment and let me know any interesting variations.

My implementation of the following paper:

Blundell et. al., Weight Uncertainty in Neural Networks, 
https://arxiv.org/pdf/1505.05424.pdf
