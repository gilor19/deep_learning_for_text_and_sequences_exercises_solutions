import numpy as np


def softmax(x):
    """
    Compute the softmax vector.
    x: an n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    z = x - x.max()     # For numeric stability
    return np.exp(z)/np.exp(z).sum()


def classifier_output(x, params):
    """
    Return the output layer (class probabilities) of a multi-layer perceptron
    Args:
        x: a vector of shape (in_dim,)
        params: a list of the form [W, b, U, b_tag]
    returns:
        probs: a vector of shape (out_dim,)
    """
    W, b, U, b_tag = params
    probs = np.tanh(x @ W + b) @ U + b_tag

    return softmax(probs)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params

    h2 = classifier_output(x, params)
    loss = -1 * np.log(h2[y])

    h2[y] -= 1
    dl_dz_2 = h2
    dz2_dh1 = U

    dh1_dz1 = 1 - np.power(np.tanh(x @ W + b), 2)

    dz1_db = 1
    gb = dz2_dh1 @ dl_dz_2 * dh1_dz1 * dz1_db

    gW = np.outer(x, gb)

    dl_db_tag = dl_dz_2 * 1
    gb_tag = dl_db_tag

    gU = np.outer(np.tanh(x @ W + b), dl_dz_2)

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """

    # start the weights with small random values sqrt(6) / sqrt(in_dim + out_dim)
    W = init_weights(hid_dim, in_dim)
    b = init_weights(hid_dim)

    U = init_weights(out_dim, hid_dim)
    b_tag = init_weights(out_dim)
    params = [W, b, U, b_tag]
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

    W, b, U, b_tag = create_classifier(3, 2, 4)

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W, U, b
        loss,grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]

    for _ in range(10):
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)
