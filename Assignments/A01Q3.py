import numpy
import math


def normalValue(mean, deviation, x):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-(x * x) / 2)


def fullGradientDescent(x_list, y_list, alpha):  # hypothesis class y is of the form w1.x1 + w2.x2 + ... + w10.x10 + c
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c = 1
    y_predicted = []
    loss = 0.0
    previous_loss = 0.0
    done = False
    n = 1
    for i in range(len(y_list)):
        y_predicted.append(0)
    while (done == False):
        for i in range(len(y_list)):  # calculating predicted values of y
            y_predicted[i] = numpy.dot(w, x_list[i]) + c
        previous_loss = loss
        for i in range(len(y_list)):
            loss += (y_list[i] - y_predicted[i]) ** 2
        loss /= len(x_list)
        """
        if len(previous_losses)<5:
            previous_losses.append(loss)
        else:
            done=math.isclose(previous_losses[0],previous_losses[1]) and math.isclose(previous_losses[1],previous_losses[2]) \
                    and math.isclose(previous_losses[2], previous_losses[3]) and math.isclose(previous_losses[5], previous_losses[4]) \
                    or loss<0.00000000000000013
        """
        done = math.isclose(loss, previous_loss,
                            abs_tol=0.0000013)  # stop criterion - comparing last two loss values to check saturation
        _w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for k in range(len(_w)):
            temp = 0.0
            for i in range(len(x_list)):
                temp += x_list[i][k] * (y_list[i] - y_predicted[i])
            _w[k] = w[k] + 2 * alpha / len(x_list) * temp
        temp = 0.0
        for i in range(len(x_list)):
            temp += (y_list[i] - y_predicted[i])
        c += 2 * alpha / len(x_list) * temp
        for k in range(len(w)):
            w[k] = _w[k]
        print("iteration ", n, "loss ", loss)
        n+=1
    print("final iteration ", n - 1, "loss ", loss, "weights ", w, "bias constant ", c)


def stochasticGradientDescent(x_list, y_list, alpha):
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c = 1
    y_predicted = []
    loss = 0.0
    previous_loss = 0.0
    done = False
    n = 1
    for i in range(len(y_list)):
        y_predicted.append(0)
    while (done == False):
        for i in range(len(y_list)):  # calculating predicted values of y
            y_predicted[i] = numpy.dot(w, x_list[i]) + c
        previous_loss = loss
        for i in range(len(y_list)):
            loss += (y_list[i] - y_predicted[i]) ** 2
        loss /= len(x_list)
        done = math.isclose(loss, previous_loss,
                            abs_tol=0.0000013)  # stop criterion - comparing last two loss values to check saturation
        for i in range(len(x_list)):
            _w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(len(_w)):
                temp = 0.0
                temp = x_list[i][k] * (y_list[i] - y_predicted[i])
                _w[k] = w[k] + 2 * alpha * temp
            temp = 0.0
            temp = (y_list[i] - y_predicted[i])
            c += 2 * alpha * temp
            for k in range(len(w)):
                w[k] = _w[k]
        print("iteration ", n, "loss ", loss)
        n+=1
    print("final iteration ", n - 1, "loss ", loss, "weights ", w, "bias constant ", c)


x_list = numpy.random.uniform(0, 1, (100, 10))
y_list = []
for i in x_list:
    y = 0
    for j in range(len(i)):
        y += (j * i[j] + 0.1 * normalValue(0, 1, j))
    y_list.append(y)

"""
run the following functions to perform linear regression with full and stochastic gradient descent respectively
"""

#fullGradientDescent(x_list,y_list,alpha=0.13)
#stochasticGradientDescent(x_list, y_list, alpha=0.001)

"""
parameter weights and bias constants were all initialized with zero in both cases

learning rate calculated with trial and error - started with 0.5 and realized that the loss was increasing with each
iteration so decreased and reached to 0.1 at which point loss started decreasing. Then again started trying for 0.19 and
reached till 0.13 when loss started decreasing again.

learning rate for the stochastic approach was decided in the same way
"""
