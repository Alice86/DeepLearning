from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[cache < 0] = 0
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']    
    xs = np.pad(x,((0,),(0,),(pad,),(pad,)),'constant',
                constant_values=(0, 0))
    
    H_ = int(1 + (H + 2 * pad - HH) / stride)
    W_ = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N, F, H_, W_))
    
    for n in range(N):
        for f in range(F):
            for h in range(H_):
                for l in range(W_):
                    out[n,f,h,l] = np.sum(xs[n,:,h*stride:h*stride+HH,
                           l*stride:l*stride+WW] * w[f,:,:,:]) + b[f]
#    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_, W_ = dout.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    xs = np.pad(x,((0,),(0,),(pad,),(pad,)),'constant',constant_values=(0, 0))
    
    dw = np.zeros((F,C,HH,WW))
    for f in range(F):
        for c in range(C):
            for h in range(HH):
                for l in range(WW):
                    dw[f,c,h,l] = np.sum(xs[:,c,h:h+H_*stride:stride,
                           l:l+W_*stride:stride]*dout[:,f,:,:])
        
    db = np.zeros((F))
    for f in range(F):
        db[f] = np.sum(dout[:,f,:,:])
    
    dxs = np.zeros((N, C, H+2*pad, W+2*pad))
    for n in range(N):
        for f in range(F):
            for h in range(H_):
                for l in range(W_):
                    dxs[n,:,h:h+HH,l:l+WW] += dout[n,f,h,l]*w[f,:,:,:]
    dx = dxs[:,:,pad:H+pad,pad:W+pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_ = int((H-Hp)/stride+1)
    W_ = int((W-Wp)/stride+1)
    
    out = np.zeros((N, C, H_, W_))
    for n in range(N):
        for c in range(C):
            for h in range(H_):
                for l in range(W_):
                    out[n,c,h,l] = np.max(x[n,c,h*stride:h*stride+Hp,l*stride:l*stride+Wp])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_ = int((H-Hp)/stride+1)
    W_ = int((W-Wp)/stride+1)

    dx = np.zeros((N, C, H, W))
    for n in range(N):
        for c in range(C):
            for h in range(H_):
                for l in range(W_):
                    maxx = np.max(x[n,c,h*stride:h*stride+Hp,l*stride:l*stride+Wp])
                    for i in range(H):
                        for j in range(W):
                            if x[n,c,i,j]==maxx:
                                dx[n,c,i,j]= dout[n,c,h,l]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
