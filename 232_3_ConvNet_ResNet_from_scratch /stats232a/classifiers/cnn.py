from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.fast_layers import *
from stats232a.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - fc - relu - fc - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final fc layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden fc layer, and keys 'W3' and 'b3' for the weights and biases       #
        # of the output fc layer.                                                  #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, 
                       (num_filters, input_dim[0], filter_size, filter_size))
        H_ = int((1 + (input_dim[1] + 2 * ((filter_size - 1) // 2) - filter_size)) / 2)
        W_ = int((1 + (input_dim[2] + 2 * ((filter_size - 1) // 2) - filter_size)) / 2)
        self.params['b1'] = np.zeros(num_filters)  
        self.params['W2'] = np.random.normal(0, weight_scale, 
                       (num_filters*H_*W_, hidden_dim))  
        self.params['b2'] = np.zeros(hidden_dim)  
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))  
        self.params['b3'] = np.zeros(num_classes)  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        outl = out1.reshape(out1.shape[0], np.prod(out1.shape[1:4]))
        out2, cache2 = fc_relu_forward(outl, W2, b2)
        out3, cache3 = fc_forward(out2, W3, b3)
        scores = out3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2)
                +np.sum(self.params['W2']**2)+np.sum(self.params['W3']**2))  
        
        dx3, dw3, grads['b3'] = fc_backward(dout, cache3)
        grads['W3'] = dw3 + self.reg * self.params['W3']

        dx2, dw2, grads['b2'] = fc_relu_backward(dx3, cache2)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        
        dout2 = dx2.reshape(out1.shape)
        dx1, dw1, grads['b1'] = conv_relu_pool_backward(dout2, cache1)
        grads['W1'] = dw1 + self.reg * self.params['W1']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
