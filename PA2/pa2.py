import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# method to import the data and return a list of data and a legend
def importData():
    data = []
    legend = []
    with open('irisdata.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            row_data = []
            # if it is the first line, use it as legend
            if row[0] == 'sepal_length':
                legend.append(row)
            else:
                # import numeric data
                for i in range(4):
                    row_data.append(float(row[i]))
                # import species data
                row_data.append(row[4])
                data.append(row_data)
        return data, legend[0]


# classify the given data with the given data type
def classifyData(data, data_type):
    filtered = []
    for row in data:
        if row[4] == data_type:
            filtered.append(row)
    return filtered


# plot the given iris data into different classes
def plotData(data):
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plot_style = ['g*', 'b+', 'rx']
    for row in data:
        if row[4] == 'setosa':
            plt.plot(row[2], row[3], plot_style[0], label=row[4])
        if row[4] == 'versicolor':
            plt.plot(row[2], row[3], plot_style[1], label=row[4])
        if row[4] == 'virginica':
            plt.plot(row[2], row[3], plot_style[2], label=row[4])


# calculate z given the weights, the data and the bias values
def logistics_func(w1, w2, x1, x2, b):
    return w1*x1 + w2*x2 + b


# calculate the sigmoid function given z
def sigmoid_func(z):
    return 1 / (1 + (np.exp(-z)))


# plot the decision boundary with the given weights and bias values
def decisionBoundary(w1, w2, b):
    x = np.arange(3, 7, 0.1)
    plt.plot(x, -(w1 / w2 * x) - (b / w2), 'k', label='Decision Boundary')


# plot the decision boundary and the given data according to the decision boundary defined
def plotDecisionBoundaryData(w1, w2, b, versicolor_data, virginica_data):
    for row in versicolor_data:
        # if it is lower or equal than the boundary, classify it as versicolor
        if logistics_func(w1, w2, row[2], row[3], b) <= 0:
            plt.plot(row[2], row[3], 'b+', label='versicolor')
        # if it is higher than the boundary, classify it as virginica
        if logistics_func(w1, w2, row[2], row[3], b) > 0:
            plt.plot(row[2], row[3], 'rx', label='virginica')

    for row in virginica_data:
        # if it is lower or equal than the boundary, classify it as versicolor
        if logistics_func(w1, w2, row[2], row[3], b) <= 0:
            plt.plot(row[2], row[3], 'b+', label='versicolor')
        # if it is higher than the boundary, classify it as virginica
        if logistics_func(w1, w2, row[2], row[3], b) > 0:
            plt.plot(row[2], row[3], 'rx', label='virginica')
    decisionBoundary(w1, w2, b)


# function for question 2 part a
# with three arguments: the data vectors, the parameters defining the neural network;
# and the pattern class
def meanSquaredError(data, weights, pattern):
    # convert all Versicolor into float values
    versicolor_data = np.array(classifyData(data, pattern[0]))[:, 2:4].astype(np.float)
    # do the same to Virginica data
    virginica_data = np.array(classifyData(data, pattern[1]))[:, 2:4].astype(np.float)
    # retrieve the weights w1 and w2, which is stored as an array in decBound[0]
    w1 = weights[0]
    w2 = weights[1]
    # retrieve the bias b, which is stored in decBound[1]
    b = weights[2]
    # expected values for versicolor (0) and virginica (1)
    y = np.append(np.zeros(len(versicolor_data)), np.ones(len(virginica_data)))
    # combined the data to a list
    combined_data = np.append(versicolor_data, virginica_data, axis=0)
    # calculate p (class probabilities) from teh combined data list
    p = np.array([sigmoid_func(logistics_func(w1, w2, i[0], i[1], b)) for i in combined_data])
    mse = ((p-y)**2).mean()
    return mse


# helper function to create a matrix from the input data
def createDataMatrix(dataList):
    output = []
    for i in range(len(dataList)):
        row = [1, dataList[i][0], dataList[i][1]]
        output.append(row)
    return np.array(output).astype(np.float)


# retrieve the gradient steps values out of the returned matrix
def getGradient(gradient_vector):
    bias_step = gradient_vector[0, 0]
    w1_step = gradient_vector[0, 1]
    w2_step = gradient_vector[0, 2]
    weights_step = np.array([w1_step, w2_step])
    return weights_step, bias_step


# function for question 2 part 2
# calculate the summed gradient for an ensemble of patterns
def gradientDescent(data, weights, pattern, stepsize):
    # convert all Versicolor into float values
    versicolor_data = np.array(classifyData(data, pattern[0]))[:, 2:4].astype(np.float)
    # do the same to Virginica data
    virginica_data = np.array(classifyData(data, pattern[1]))[:, 2:4].astype(np.float)
    # retrieve the weights w1 and w2, which is stored as an array in decBound[0]
    w1 = weights[0]
    w2 = weights[1]
    # retrieve the bias b, which is stored in decBound[1]
    b = weights[2]
    # combined the two data set into one array
    combined_data = np.append(versicolor_data, virginica_data, axis=0)
    # create a matrix storing input data
    x = createDataMatrix(combined_data)
    # expected values for versicolor (0) and virginica (1)
    y = np.append(np.zeros(len(versicolor_data)), np.ones(len(virginica_data)))
    # class probabilities: calculated sigmoid values from the input data
    p = np.array([sigmoid_func(logistics_func(w1, w2, i[0], i[1], b)) for i in combined_data])
    # total number of input data rows
    n = len(combined_data)
    # calculate the gradient descent according to the function derived in the writeup Q3C
    gradient = stepsize * (2 * np.matrix((np.ones(len(p)) - p) * (p - y)) * x/n)
    # gradient returns a matrix, use helper function getGradient to retrieve the values
    return getGradient(gradient)


# Q3A,B,C
# create 3 sets of plots that represent the starting, the mid-way and the final decBound plot and the learning curves
def plotGradientDescent(plotnum, weights, bias, data, stepsize):
    # ***SETUP***
    # setup the data for calculating gradient descent
    versicolor_data = classifyData(data, 'versicolor')
    virginica_data = classifyData(data, 'virginica')
    x = np.arange(3, 7, 0.1)
    w = weights
    b = bias
    y = (-w[0] / w[1]) * x + (-b / w[1])
    # the number of steps taken
    iteration = 0
    # list that stores the MSE for at gradient step
    mse = [meanSquaredError(data, [w[0], w[1], b], ('versicolor', 'virginica'))]
    print('Starting decision boundary = y = {}*x + {}'.format((-w[0] / w[1]), (-b / w[1])))
    print('Starting MSE: {}'.format(mse[0]))
    # **Plot Setup**
    # Starting DecBound Plot
    plt.figure(plotnum)
    plt.subplot(121)
    plt.title('Starting DecBound')
    plotData(versicolor_data)
    plotData(virginica_data)
    plt.plot(x, y, color='k')
    legend = [Line2D([0], [0], color='k', label='Iteration {}'.format(iteration))]
    plt.legend(handles=legend, loc='lower right')
    # Starting Learning Curve
    plt.subplot(122)
    plt.title('Starting Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.plot(mse, 'k')

    # ***STEPPING***
    while mse[-1] > 0.04:
        weight_step, bias_step = gradientDescent(data, [w[0], w[1], b], ('versicolor', 'virginica'), stepsize)
        # update the weights and bias
        w = np.array(w - weight_step)
        b -= bias_step
        # print(mse[-1])
        # append the mse
        mse.append(meanSquaredError(data, [w[0], w[1], b], ('versicolor', 'virginica')))
        # increment the iteration
        iteration += 1
        # Draw a plot in the middle
        if iteration == 8000:
            # Mid-way DecBound Plot
            plt.figure(plotnum+1)
            plt.subplot(121)
            plt.title('Mid-way DecBound')
            y = (-w[0] / w[1]) * x + (-b / w[1])
            plotData(versicolor_data)
            plotData(virginica_data)
            plt.plot(x, y, color='k')
            legend = [Line2D([0], [0], color='k', label='Iteration {}'.format(iteration))]
            plt.legend(handles=legend)
            # Mid-way Learning Curve
            plt.subplot(122)
            plt.title('Mid-way Learning Curve')
            plt.xlabel('Iterations')
            plt.ylabel('MSE')
            plt.plot(mse, 'k')

    # ***ENDING***
    print('Ending decision boundary = y = {}*x + {}'.format((-w[0] / w[1]), (-b / w[1])))
    print('Final MSE: {}'.format(mse[-1]))
    print('Total Iterations : {}'.format(iteration))
    # Final DecBound Plot
    plt.figure(plotnum+2)
    plt.subplot(121)
    plt.title('Final DecBound'.format(iteration))
    y = (-w[0] / w[1]) * x + (-b / w[1])
    plotData(versicolor_data)
    plotData(virginica_data)
    plt.plot(x, y, color='k')
    legend = [Line2D([0], [0], color='k', label='Iteration {}'.format(iteration))]
    plt.legend(handles=legend)
    # Final Learning Curve
    plt.subplot(122)
    plt.title('Final Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.plot(mse, 'k')
    plt.show()


def main():
    data, legend = importData()
    setosa_data = classifyData(data, 'setosa')
    versicolor_data = classifyData(data, 'versicolor')
    virginica_data = classifyData(data, 'virginica')
    legends = [Line2D([0], [0], marker='+', color='b', linestyle='None',
                      label='Versicolor', markersize=10),
               Line2D([0], [0], marker='x', color='r', linestyle='None',
                      label='Virginica', markersize=10)]

    # q1a
    plt.figure(1)
    plt.title('Q1A: Versicolor and Virginica Plot')
    plotData(versicolor_data)
    plotData(virginica_data)
    plt.legend(handles=legends, loc='upper left')

    # q1b Output Function
    # z = w1x1 + w2x2 + b
    # more detailed in writeup

    # q1c
    plt.figure(2)
    plt.title('Q1C: Versicolor and Virginica Decision Boundary')
    plotData(versicolor_data)
    plotData(virginica_data)
    decisionBoundary(1.8, 4, -15.3)
    plt.legend(handles=legends, loc='upper left')

    # q1d
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Q1D: Output of your neural network over the input space')
    x = np.arange(3, 7, 0.1)
    y = np.arange(0.75, 2.75, 0.1)
    X, Y = np.meshgrid(x, y)
    w1 = 1.8
    w2 = 4
    b = -15.3
    zs = np.array([sigmoid_func(logistics_func(w1, w2, x, y, b)) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k')
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Sigmoid (z)')

    # q1e
    plt.figure(4)
    plt.title('Q1E')
    plotDecisionBoundaryData(1.8, 4, -15.3, versicolor_data, virginica_data)
    plt.legend(handles=legends, loc='upper left')

    # q2a
    # function meanSquaredError gives the mse.

    # q2b
    # setting m1 = 1.8, m2 = 4, b = -15.3 yield a relatively small error
    plt.figure(5)
    plt.title('Q2B-1: Graph with decision boundary of small error')
    w1 = 1.8
    w2 = 4
    b = -15.3
    plotData(versicolor_data)
    plotData(virginica_data)
    decisionBoundary(w1, w2, b)
    mse = meanSquaredError(data, [w1, w2, b], ('versicolor', 'virginica'))
    plt.legend(handles=legends, loc='upper left')
    print('Small MSE = ' + str(mse))
    plt.figure(6)
    plt.title('Q2B-2: Graph with decision boundary of large error')
    w1 = 1.8
    w2 = 1.8
    b = 15.3
    plotData(versicolor_data)
    plotData(virginica_data)
    decisionBoundary(w1, w2, b)
    plt.legend(handles=legends, loc='upper left')
    mse = meanSquaredError(data, [w1, w2, b], ('versicolor', 'virginica'))
    print('Large MSE = ' + str(mse))

    # q2c and d in writeup

    # q2e
    plt.figure(7)
    plt.title('Q2E: Gradient descent with stepsize 0.1')
    plotData(versicolor_data)
    plotData(virginica_data)
    w1 = 1.8
    w2 = 4
    w = np.array([w1, w2])
    b = -15.3
    decisionBoundary(w1, w2, b)
    plt.legend(handles=legends, loc='upper left')
    weight_step, bias_step = gradientDescent(data, [w[0], w[1], b], ('versicolor', 'virginica'), 0.1)
    w = np.array(w - weight_step)
    b -= bias_step
    x = np.arange(3, 7, 0.1)
    y = -(w[0] / w[1]) * x - (b / w[1])
    plt.plot(x, y, color='green')

    # q3a and b
    # using w1 = 1.8, w2 = 4, b = -15.3
    plotGradientDescent(8, [1.5, 2.5], -15.5, data, 0.1)

    # q3c
    # using controlled randomized numbers
    w1 = random.uniform(0, 2)
    w2 = random.uniform(2, 4)
    b = -random.uniform(10, 20)
    w = [w1, w2]
    print('\nQ3c Random Numbers')
    print('Starting weights: w1 = {}, w2 = {}, b = {}'.format(w1, w2, b))
    plotGradientDescent(11, w, b, data, 0.1)

    plt.show()


if __name__ == '__main__':
    main()


