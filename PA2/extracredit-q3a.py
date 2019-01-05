import numpy
import pandas
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split


def trainFor(epochs):
    dataBase = pandas.read_csv("irisdata.csv", header=None)
    dataSet = dataBase.values
    Input = dataSet[51:, 2:4].astype(float)
    Output = dataSet[51:, 4]
    encodeOutput = []
    for i in Output:
        if i == 'versicolor':
            encodeOutput.append(0)
        else:
            encodeOutput.append(1)
    plt.scatter(Input[0:50, 0], Input[0:50, 1], c='b', label='Versicolor')
    plt.scatter(Input[50:100, 0], Input[50:100, 1], c='r', label='Virginica')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.savefig('q4a-2classes.png')
    plt.show()

    inputTrain, inputVal, outputTrain, outputVal = train_test_split(Input, encodeOutput, test_size=0.25, shuffle=True)
    model = modelNN()
    model.fit(x=inputTrain, y=outputTrain, epochs=epochs, validation_data=(inputVal, outputVal))


def modelNN():
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    return model


def main():
    trainFor(2000)
    trainFor(4000)


if __name__ == '__main__':
    main()
