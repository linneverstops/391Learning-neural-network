import numpy
import pandas
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split


def trainWithSepalAndPetal():
    dataBase = pandas.read_csv("irisdata.csv", header=None)
    dataSet = dataBase.values

    Input = dataSet[1:, 0:4].astype(float)
    Output = dataSet[1:, 4]
    encodeOutput = []
    for i in Output:
        if i == 'setosa':
            encodeOutput.append(0)
        else:
            if i == 'versicolor':
                encodeOutput.append(1)
            else:
                encodeOutput.append(2)

    plt.figure(1)
    plt.title('Scatter Plot using Petal data')
    plt.scatter(Input[0:50, 2], Input[0:50, 3], c='b', label='Setosa')
    plt.scatter(Input[50:100, 2], Input[50:100, 3], c='r', label='Versicolor')
    plt.scatter(Input[100:150, 2], Input[100:150, 3], c='g', label='Virginica')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.savefig('q4b-3classes-petal.png')

    # showing the variance between all 3 classes using sepal lengths and widths
    plt.figure(2)
    plt.title('Scatter Plot using Sepal data')
    plt.scatter(Input[0:50, 0], Input[0:50, 1], c='b', label='Setosa')
    plt.scatter(Input[50:100, 0], Input[50:100, 1], c='r', label='Versicolor')
    plt.scatter(Input[100:150, 0], Input[100:150, 1], c='g', label='Virginica')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(loc='upper left')
    plt.savefig('q4b-3classes-sepal.png')
    plt.show()

    inputTrain, inputVal, outputTrain, outputVal = train_test_split(Input, encodeOutput, test_size=0.25, shuffle=True)
    model = modelNN(4)
    model.fit(x=inputTrain, y=outputTrain, epochs=2000, validation_data=(inputVal, outputVal))


def modelNN(dimensions):
    model = Sequential()
    model.add(Dense(1, input_dim=dimensions, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    return model


def main():
    trainWithSepalAndPetal()


if __name__ == '__main__':
    main()
