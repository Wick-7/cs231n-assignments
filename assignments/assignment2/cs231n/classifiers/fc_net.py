from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        for i in range(self.num_layers-1):
          if i == 0:
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
          else:
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
          self.params[f'b{i+1}'] = np.zeros(hidden_dims[i])
          if self.normalization == 'batchnorm':
            self.params[f'gamma{i+1}'] = np.ones(hidden_dims[i])
            self.params[f'beta{i+1}'] = np.zeros(hidden_dims[i])
        self.params[f'W{self.num_layers}'] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes)
                                                                     
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        N = X.shape[0]
        out = X.reshape(N, -1)
        aff_cache, nor_cache, relu_cache, drop_cache = [], [], [], []
        L = self.num_layers
        for i in range(1, L):
          W_i = self.params[f'W{i}']
          b_i = self.params[f'b{i}']
          aff_cache.append((out, W_i, b_i))
          out = out.dot(W_i) + b_i
          if self.normalization == 'batchnorm':
            gamma_i = self.params[f'gamma{i}']
            beta_i = self.params[f'beta{i}']
            bn_param = self.bn_params[i-1]
            eps = bn_param.get("eps", 1e-5)
            if mode =='train':
              mean = np.mean(out, axis=0)
              var = np.var(out, axis=0)
              x_hat = (out - mean) / np.sqrt(var + eps)
              nor_cache.append((out, x_hat, mean, var, gamma_i, beta_i, eps))
              out = gamma_i * x_hat + beta_i
              momentum = 0.9
              running_mean = bn_param.get('running_mean', np.zeros_like(mean))
              running_var = bn_param.get('running_var', np.zeros_like(var))

              running_mean = momentum * running_mean + (1 - momentum) * mean
              running_var = momentum * running_var + (1 - momentum) * var

              bn_param['running_mean'] = running_mean
              bn_param['running_var'] = running_var
            else:
              running_mean = bn_param.get('running_mean', np.zeros(out.shape[1]))
              running_var = bn_param.get('running_var', np.zeros(out.shape[1]))
              x_hat = (out - running_mean) / np.sqrt(running_var + eps)
              nor_cache.append((out, x_hat, running_mean, running_var, gamma_i, beta_i, eps))
              out = gamma_i * x_hat + beta_i
          relu_cache.append(out.copy())
          out = np.maximum(0, out)
          if self.use_dropout:
            if mode == 'train':
              if 'seed' in self.dropout_param:
                np.random.seed(self.dropout_param['seed'])
              mask = (np.random.rand(*out.shape) < self.dropout_param['p']) / self.dropout_param['p']
              drop_cache.append(mask)
              out = out * mask
        W_L = self.params[f'W{L}']
        b_L = self.params[f'b{L}']
        aff_cache.append((out, W_L, b_L))
        scores = out.dot(W_L) + b_L
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.sum(np.log(p[np.arange(N), y]))
        loss = loss / N + np.sum([0.5 * self.reg * np.sum(self.params[f'W{i}']**2) for i in range(1, L+1)])
        dscores = p
        dscores[np.arange(N), y] -= 1
        dscores /= N
        X_L, W_L, b_L = aff_cache[-1]
        grads[f'b{L}'] = np.sum(dscores, axis=0)
        grads[f'W{L}'] = X_L.T.dot(dscores) + self.reg * W_L
        dx = dscores.dot(W_L.T)
        for i in reversed(range(1, L)):
          if self.use_dropout and mode=='train':
            mask = drop_cache[i-1]
            dx = dx * mask
          dx = dx * (relu_cache[i-1]>0)
          if self.normalization == 'batchnorm':
            x, x_hat, mean, var, gamma_i, beta_i, eps = nor_cache[i-1]
            N_bn = x.shape[0]
            grads[f'beta{i}'] = np.sum(dx, axis=0)
            grads[f'gamma{i}']= np.sum(dx * x_hat, axis=0)
            dx_hat = dx * gamma_i
            dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
            dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
            dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N_bn + dmean / N_bn
          X_i, W_i, b_i = aff_cache[i-1]
          grads[f'b{i}'] = np.sum(dx, axis=0)
          grads[f'W{i}'] = X_i.T.dot(dx) + self.reg * W_i
          dx = dx.dot(W_i.T)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
