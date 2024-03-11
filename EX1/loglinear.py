import numpy as np


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    z = x - x.max() # For numeric stability
    return np.exp(z)/np.exp(z).sum()
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    probs = softmax((x @ W) + b)
    return probs


def predict(x, params):
    """
    Returns the prediction (the highest scoring class id) of a
    log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    probs = classifier_output(x, params)
    loss = -1 * np.log(probs[y])

    # compute gradients
    # z = Wx + b
    # in the case where i = k
    # dL / db = (dz / db) * (dL / dz)
    # [dz / db = d(Wx + b) / db = d(Wx) / db + d(b) / db = 0 + 1 = 1]

    # dl / db = 1 * (dL / dz) = -log(y_hat) / dz

    # the derivative of y_hat = softmax(wx+b) = y_hat * (1 - y_hat)
    # (-log(yhat) / dz) =[chain rule] (-1 / y_hat) * y_hat * (1 - y_hat) = y_hat - 1
    # Therefore dl / db = y_hat - 1

    # dl / dw = dl / dz * dz / dw
    # dz/dw = d(wx+b)/dw = d(wx)/dw + d(b)/dw = x + 0 = x

    # dl / dw = x * (dL / dz) = x*(y_hat - 1)
    # Therefore dl/dw = x*(y_hat - 1)

    gb = probs.copy()
    gb[y] -= 1

    # to compute the gradient of W, we need to take the outer product of x and the gradient of b
    # this is because dl/dw = x * dl/dz = x * 1*dl/db = x * (y_hat - 1) = x * gb

    gW = np.outer(x, gb)
    return loss, [gW, gb]


def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    # start the weights with small random values sqrt(6) / sqrt(in_dim + out_dim)
    W = np.random.uniform(-np.sqrt(6) / np.sqrt(in_dim + out_dim),
                          np.sqrt(6) / np.sqrt(in_dim + out_dim),
                          (in_dim, out_dim))
    b = np.random.uniform(-np.sqrt(6) / np.sqrt(in_dim + out_dim),
                          np.sqrt(6) / np.sqrt(in_dim + out_dim),
                          out_dim)
    return [W, b]


if __name__ == '__main__':
    # Sanity checks for softmax.
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss,grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
