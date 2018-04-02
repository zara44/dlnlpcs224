import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, dimensions, params):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
        Arguments:

    data -- M x Dx matrix, where each row is a training example.

    labels -- M x Dy matrix, where each row is a one-hot vector.

    params -- Model parameters, these are unpacked for you.

    dimensions -- A tuple of input dimension, number of hidden units

                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:(ofs+ Dx * H)], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    print "running forward"
    ### YOUR CODE HERE: forward propagation
    # print(np.dot(data,W1)+b1)
    h = sigmoid(np.dot(data,W1)+b1)
    y_hat = softmax(np.dot(h,W2)+b2)
    # raise NotImplementedError
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    cost = np.sum(-np.log(y_hat[labels==1])) / data.shape[0]

    d3 = (y_hat - labels) / data.shape[0]

    gradW2 = np.dot(h.T, d3)

    gradb2 = np.sum(d3,0,keepdims=True)

    dh = np.dot(d3,W2.T)

    grad_h = sigmoid_grad(h) * dh

    gradW1 = np.dot(data.T,grad_h)

    gradb1 = np.sum(grad_h,0)
    # raise NotImplementedError
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."
    dimensions = [10, 5, 10]
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    # def temp_func(params, dimensions=dimensions):


    N = 20

    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1



    #     return forward_backward_prop(data, labels, dimensions, params=params)
    #
    #
    # gradcheck_naive(temp_func(params),params[0])
    # f= lambda x:forward_backward_prop(data, labels, dimensions, x)
    gradcheck_naive(lambda x:forward_backward_prop(data, labels, dimensions, x), params)

    return

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()