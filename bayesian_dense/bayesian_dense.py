from keras.layers.core import Layer
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec, Layer, Merge
from keras.regularizers import ActivityRegularizer, Regularizer


class VariationalRegularizer(Regularizer):
    """Regularization layer for BayesianDense Layer Weights"""
    def __init__(self, weight=1e-3):
        Regularizer.__init__(self)
        self.weight=weight
        self.uses_learning_phase=True

    def __call__(self, loss):
        reg = - 0.5 * K.mean(1 + self.p - K.exp(self.p), axis=None)
        return K.in_train_phase(loss + self.weight*reg, loss)

    def get_config(self):
        config = {"weight":float(self.weight)}
        base_config = super(VariationalRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def init_uniform(shape, lb, ub, name=None):
    return K.random_uniform_variable(shape, lb, ub, name=name)

class BayesianDense(Layer):
    """Bayesian layer as described in https://arxiv.org/abs/1505.05424
    # Example
    ```python
        layer = BayesianDense(256,
            W_sigma_regularizer=VariationalRegularizer(1e-3),
            b_sigma_regularizer=VariationalRegularizer(1e-3),
            W_regularizer=le(1e-3))
    ```
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    """
    def __init__(self, output_dim, init='glorot_uniform',
                 init_sigma=lambda shape, name:init_uniform(shape, -10, -5, name=name),
                 activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, W_sigma_regularizer=None, b_sigma_regularizer=None,
                 activity_regularizer=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.init_sigma = init_sigma
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.uses_learning_phase = True

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_sigma_regularizer = regularizers.get(W_sigma_regularizer)
        self.b_sigma_regularizer = regularizers.get(b_sigma_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(BayesianDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_dim = input_shape[1]
        input_dim=self.input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W_mu = self.init((input_dim, self.output_dim),
                           name='{}_W_mu'.format(self.name))
        self.W_log_sigma = self.init_sigma((input_dim, self.output_dim),
                           name='{}_W_log_sigma'.format(self.name))
        if self.bias:
            self.b_mu = self.init((self.output_dim,),
                             name='{}_b_mu'.format(self.name))
            self.b_log_sigma = self.init_sigma((self.output_dim,),
                             name='{}_b_log_sigma'.format(self.name))
            self.trainable_weights = [self.W_mu, self.W_log_sigma, self.b_mu, self.b_log_sigma]
        else:
            self.trainable_weights = [self.W_mu, self.W_log_sigma]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W_mu)
            self.regularizers.append(self.W_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b_mu)
            self.regularizers.append(self.b_regularizer)
        if self.W_sigma_regularizer:
            self.W_sigma_regularizer.set_param(self.W_log_sigma)
            self.regularizers.append(self.W_sigma_regularizer)
        if self.b_sigma_regularizer:
            self.b_sigma_regularizer.set_param(self.b_log_sigma)
            self.regularizers.append(self.b_sigma_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        e = K.random_normal((x.shape[0], self.input_dim, self.output_dim))
        w = self.W_mu.dimshuffle('x',0,1)+e*(K.exp(self.W_log_sigma/2).dimshuffle('x',0,1))
        output = K.batch_dot(x, w)
        test_output = K.dot(x, self.W_mu)
        if self.bias:
            eb = K.random_normal((x.shape[0], self.output_dim))
            b = self.b_mu.dimshuffle('x',0)+eb*(K.exp(self.b_log_sigma/2).dimshuffle('x',0))
            output += b
            test_output += self.b_mu

        return self.activation(K.in_train_phase(output, test_output))

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_sigma_regularizer': self.W_sigma_regularizer.get_config() if self.W_sigma_regularizer else None,
                  'b_sigma_regularizer': self.b_sigma_regularizer.get_config() if self.b_sigma_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(BayesianDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
