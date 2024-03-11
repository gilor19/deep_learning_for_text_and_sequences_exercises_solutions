import numpy as np
from typing import Callable


def gradient_check(f: Callable, x: np.ndarray) -> None:
    """ 
    Gradient check for a function f

    Args:
        f (Callable) should be a function that takes a single argument and outputs the tuple (loss, grad)
        x (numpy array) is the point to check the gradient at

    Raises:
        AssertionError if the numerical and analytic gradients are different
    """ 
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # epsilon

    # Iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        loss_idx = 0
        x_plux = x.copy()
        x_minus = x.copy()
        x_plux[ix] = x_plux[ix] + h
        x_minus[ix] = x_minus[ix] - h
        numeric_gradient = (f(x_plux)[loss_idx] - f(x_minus)[loss_idx]) / (2 * h)

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return
    
        it.iternext()  # Step to next index

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))       # scalar test
    gradient_check(quad, np.random.randn(3,))     # 1-D test
    gradient_check(quad, np.random.randn(4, 5))   # 2-D test
    print()


if __name__ == '__main__':
    sanity_check()
