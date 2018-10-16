from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.fast_layers import *
from stats232a.layer_utils import *

class OneBlockResnet(object):
    """
    A convolutional network with a residual block:
    conv - relu - 2x2 max pool - residual block - relu - fc - relu - fc - softmax
    The residual block has the following structure:
           ______________________      
          |                      |
    input - conv - relu - conv - + - output
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[64, 64, 64], filter_size=[7, 3, 3], 
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in each convolutional layer.
        - filter_size: Size of filters to use in each convolutional layer.
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
        # TODO: Initialize weights and biases for the one block residual network.  #
        # Weights should be initialized from a Gaussian with standard deviation    #
        # equal to weight_scale; biases should be initialized to zero.             #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2', and keys 'W3' and 'b3' for the two     #
        # convolutional layers in residual block; use keys 'W4' and 'b4' for the   #
        # hidden fc layer; use keys 'W5' and 'b5' for the output fc layer.         #
        ############################################################################
        dim = input_dim
        pool_sizes = (2, 1, 1)
        for i in range(len(num_filters)):
            f = num_filters[i]
            size = filter_size[i]
            pad = (filter_size[i] - 1) // 2
            self.params['W'+str(i+1)] = np.random.normal(0, weight_scale, 
                       (f, dim[0], size, size))
            H_ = int((1 + (dim[1] + 2 * pad - size))/pool_sizes[i])
            W_ = int((1 + (dim[2] + 2 * pad - size))/pool_sizes[i])
            dim = (f, H_, W_)
            self.params['b'+str(i+1)] = np.zeros(f)  
        
        self.params['W4'] = np.random.normal(0, weight_scale, 
                       (np.prod(dim), hidden_dim))  
        self.params['b4'] = np.zeros(hidden_dim)  
        self.params['W5'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))  
        self.params['b5'] = np.zeros(num_classes)  

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
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W3.shape[2]
        conv_param3 = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one block residual net,         #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
        out2, cache2 = conv_relu_forward(out1, W2, b2, conv_param2)
        out3, cache3 = conv_forward_fast(out2, W3, b3, conv_param3)
        out_res = out3+out1
        out_rl, cache_rl = relu_forward(out_res)
        
        outl = out_rl.reshape(out_rl.shape[0], np.prod(out_rl.shape[1:4]))
        out4, cache4 = fc_relu_forward(outl, W4, b4)
        out5, cache5 = fc_forward(out4, W5, b5)
        
        scores = out5
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one block residual net,        #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        for i in range(5):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]**2)  
        
        dx5, dw5, grads['b5'] = fc_backward(dout, cache5)
        grads['W5'] = dw5 + self.reg * self.params['W5']

        dx4, dw4, grads['b4'] = fc_relu_backward(dx5, cache4)
        grads['W4'] = dw4 + self.reg * self.params['W4']
        
        dx_rl = relu_backward(dx4.reshape(out_rl.shape), cache_rl)
        
        dx3, dw3, grads['b3'] = conv_backward_fast(dx_rl, cache3)
        grads['W3'] = dw3 + self.reg * self.params['W3']
        
        dx2, dw2, grads['b2'] = conv_relu_backward(dx3, cache2)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        
        dx1, dw1, grads['b1'] = conv_relu_pool_backward(dx2+dx_rl, cache1)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
