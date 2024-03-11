import numpy as np


def softmax(x):
    """
    Compute the softmax vector.
    x: an n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    z = x - x.max()         # For numeric stability
    return np.exp(z)/np.exp(z).sum()


def forward_pass(x, params):
    i = 0
    z_layers = []   # z is of the form z = xW + b
    h_layers = [x]  # h is of the form h = activation(z)
    probs = x
    while i < len(params):
        weights = params[i]
        bias = params[i + 1]
        z = probs @ weights + bias
        z_layers.append(z)
        if i == len(params) - 2:
            probs = softmax(z)
        else:
            probs = np.tanh(z)
        h_layers.append(probs)
        i += 2
    return z_layers, h_layers


def classifier_output(x, params):
    i = 0
    probs = x
    while i < len(params):
        weights = params[i]
        bias = params[i+1]
        if i == len(params) - 2:
            probs = softmax(probs @ weights + bias)
        else:
            probs = np.tanh(probs @ weights + bias)
        i += 2
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    """

    h_final = classifier_output(x, params)
    loss = -1 * np.log(h_final[y])
    h_final[y] -= 1
    z_layers, h_layers = forward_pass(x, params)

    # backprop
    i = 0
    weights = params[::2]

    grads = []
    while i < len(weights):
        gb = h_final
        n = len(weights) - 1
        while n > i:
            gb = weights[n] @ gb
            d_activation = 1 - np.power(np.tanh(z_layers[n-1]), 2)
            gb = d_activation * gb
            n -= 1

        gW = np.outer(h_layers[i], gb)
        grads.append(gW)
        grads.append(gb)
        i += 1

    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    i = 0
    while i < len(dims) - 1:
        layer_out_dims = dims[i + 1]
        layer_in_dims = dims[i]
        W = init_weights(in_dim=layer_in_dims, out_dim=layer_out_dims)
        b = init_weights(out_dim=layer_out_dims)
        params.append(W)
        params.append(b)
        i += 1
    return params


def init_weights(out_dim, in_dim=0):
    if in_dim:
        size = (in_dim, out_dim)
    else:
        size = (out_dim,)
    return np.random.uniform(-np.sqrt(6) / np.sqrt(in_dim + out_dim),
                             np.sqrt(6) / np.sqrt(in_dim + out_dim),
                             size=size)


if __name__ == '__main__':
    from grad_check import gradient_check


    def _loss_and_w1_grad(w1):
        global b1, w2, b2, w3, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[0]

    def _loss_and_b1_grad(b1):
        global w1, w2, b2, w3, b3
        loss,grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[1]

    def _loss_and_w2_grad(w2):
        global w1, b1, b2, w3, b3
        loss,grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[2]

    def _loss_and_b2_grad(b2):
        global w1, b1, w2, w3, b3
        loss,grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[3]

    def _loss_and_w3_grad(w3):
        global w1, b1, w2, b2, b3
        loss,grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[4]

    def _loss_and_b3_grad(b3):
        global w1, b1, w2, b2, w3
        loss,grads = loss_and_gradients([1, 2, 3], 0, [w1, b1, w2, b2, w3, b3])
        return loss, grads[5]

    for _ in range(10):
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = create_classifier([3, 30, 40, 50, 20, 10])
        gradient_check(_loss_and_w1_grad, w1)
        gradient_check(_loss_and_b1_grad, b1)
