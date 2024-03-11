import loglinear as ll
import random
from argparse import ArgumentParser
import numpy as np
import utils


def feats_to_vec(features):
    """
    Create a bag-of-words vector representation for a sentence.
    Args:
        features: the features of a sentence.

    Returns:
        A BoW vector representation of the sentence. the vector size is the number of features.
    """
    global pre_process
    vec = np.zeros(len(pre_process.F2I))
    for f in features:
        if f in pre_process.F2I:
            vec[pre_process.F2I[f]] += 1
    return vec


def accuracy_on_dataset(dataset, params):
    """ Compute the accuracy of the current parameters on the dataset."""
    good = bad = 0.0
    for label, features in dataset:
        y_hat = ll.predict(feats_to_vec(features), params)
        if y_hat == pre_process.L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, features) pairs.
    dev_data  : a list of (label, features) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of the parameters (initial values) [W,b].
    """
    for i in range(num_iterations):
        cumulative_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = pre_process.L2I[label]         # convert the label to its index.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cumulative_loss += loss
            # update the parameters according to the gradients and the learning rate.
            W, b = params
            w_grad, b_grad = grads
            params = W - learning_rate * w_grad, b - learning_rate * b_grad

        train_loss = cumulative_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(f"iteration: {i}, train_loss: {train_loss},"
              f" train_accuracy: {train_accuracy}, dev_accuracy: {dev_accuracy}")
    return params


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", help="path to train set", default="train")
    parser.add_argument("-d", "--dev", dest="dev", help="path to dev set", default="dev")
    parser.add_argument("--test", dest="test", help="path to test set")  # optional
    args = parser.parse_args()

    pre_process = utils.PreProcess(args.train, args.dev, args.test)

    # out dim is the number of labels
    out_dim = len(pre_process.L2I)
    # in dim is the number of features
    in_dim = len(pre_process.F2I)

    # the following parameters were manually optimized
    num_iterations = 20
    learning_rate = 0.1

    params = ll.create_classifier(in_dim, out_dim)
    train_data = pre_process.train_data
    dev_data = pre_process.dev_data

    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    if args.test:
        test_vecs = [feats_to_vec(f) for f in pre_process.test_data]
        test_labels = [ll.predict(x, trained_params) for x in test_vecs]

        # convert prediction labels to language labels e.g 0 -> en
        test_labels = [pre_process.I2L[l] for l in test_labels]
        utils.write_preds(test_labels, 'loglin.test.pred')
