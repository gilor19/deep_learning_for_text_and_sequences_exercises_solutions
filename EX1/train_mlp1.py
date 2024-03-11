import mlp1
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


def accuracy_on_dataset(dataset, params_):
    """ Compute the accuracy of the current parameters on the dataset."""
    good = bad = 0.0
    for label, features in dataset:
        y_hat = mlp1.predict(feats_to_vec(features), params_)
        if y_hat == pre_process.L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data_, dev_data_, num_iterations_, learning_rate_, params_):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for i in range(num_iterations_):
        cumulative_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data_)
        for label, features in train_data_:
            x = feats_to_vec(features)  # convert features to a vector.
            y = pre_process.L2I[label]         # convert the label to its index.
            loss, grads = mlp1.loss_and_gradients(x, y, params_)
            cumulative_loss += loss
            # update the parameters according to the gradients and the learning rate.
            W, b, U, b_tag = params_
            w_grad, b_grad, u_grad, gb_tag = grads
            params_ = W - learning_rate_ * w_grad, b - learning_rate_ * b_grad,\
                U - learning_rate_ * u_grad, b_tag - learning_rate_ * gb_tag

        train_loss = cumulative_loss / len(train_data_)
        train_accuracy = accuracy_on_dataset(train_data_, params_)
        dev_accuracy = accuracy_on_dataset(dev_data_, params_)
        print(f"iteration: {i}, train_loss: {train_loss},"
              f" train_accuracy: {train_accuracy}, dev_accuracy: {dev_accuracy}")
    return params_


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", help="path to train set", default="train")
    parser.add_argument("-d", "--dev", dest="dev", help="path to dev set", default="dev")
    parser.add_argument("--test", dest="test", help="path to test set")  # optional
    args = parser.parse_args()

    pre_process = utils.PreProcess(args.train, args.dev, args.test, gram='bigram')

    # out dim is the number of labels
    out_dim = len(pre_process.L2I)
    # in dim is the number of features
    in_dim = len(pre_process.F2I)

    # the following parameters were manually optimized
    hid_dim = 50
    learning_rate = 0.05
    num_iterations = 20

    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    train_data = pre_process.train_data
    dev_data = pre_process.dev_data

    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    if args.test:
        test_vecs = [feats_to_vec(f) for f in pre_process.test_data]
        test_labels_idx = [mlp1.predict(x, trained_params) for x in test_vecs]
        # convert prediction labels to language labels e.g 0 -> en
        test_labels = [pre_process.I2L[l] for l in test_labels_idx]
        utils.write_preds(test_labels, 'mlp1.test.pred')

