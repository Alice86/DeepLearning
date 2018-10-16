pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    pass
    outs, fc = fc_forward(x, w, b)
    out, rl = relu_forward(outs)
    cache = (fc, rl)    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    pass
    fc, rl = cache      
    dout = relu_backward(dout, rl)
    dx, dw, db = fc_backward(dout, fc)   
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db
