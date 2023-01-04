import random
from matplotlib import pyplot as plt
from abc import ABC
import numpy as np
from numpy.random import permutation


class RBF:

    def __init__(self, input_dimension, number_centers, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.number_centers = number_centers
        self.centers = [np.arange(-1, 1, input_dimension) for i in range(number_centers)]
        self.beta = 8
        self.weights = random.random()

    def basis_function(self, c, d):
        assert len(d) == self.input_dimension
        return np.exp(-self.beta * np.linalg.norm(c - d) ** 2)

    def calculate_activations_rbf(self, X):
        # calculate activations of RBFs
        matrix = np.zeros((X.shape[0], self.number_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                matrix[xi, ci] = self.basis_function(c, x)
        return matrix

    def train(self, x, y):
        """ X: matrix of dimensions n x input_dimension
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = permutation(x.shape[0])[:self.number_centers]
        self.centers = [x[i, :] for i in rnd_idx]

        # calculate activations of RBFs
        matrix_g = self.calculate_activations_rbf(x)

        # calculate output weights (pseudoinverse)
        self.weights = np.dot(np.linalg.pinv(matrix_g), y)

    def test(self, x):
        """ X: matrix of dimensions n x indim """

        matrix_g = self.calculate_activations_rbf(x)
        y = np.dot(matrix_g, self.weights)
        return y


class Phi(ABC):
    def implement(self, x, **kwargs):
        pass


class Sin(Phi):
    """
    sin(x)
    """

    def implement(self, x, **kwargs):
        return np.sin(x)


class XSin(Phi):
    """
    x^2 * |sin(x * pi)|
    """

    def implement(self, x, **kwargs):
        return x ** 2 * np.absolute(np.sin(np.pi * x))


class Fit:
    def __init__(self, number_data, input_dimension, output_dimension, number_centers, from_num, to_num, data_function):
        self.number_data = number_data
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.number_centers = number_centers
        self.from_num = from_num
        self.to_num = to_num
        self.data_function = data_function

    def fit(self):
        x = np.mgrid[self.from_num:self.to_num:complex(0, self.number_data)].reshape(self.number_data, 1)
        y = self.data_function(x)
        rbf = RBF(self.input_dimension, self.number_centers, self.output_dimension)
        rbf.train(x, y)
        z = rbf.test(x)
        self.draw_plot(rbf, x, y, z)

    def draw_plot(self, rbf, x, y, z):
        # plot original data
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'k-', linewidth=2)

        # plot learned model
        plt.plot(x, z, 'r-', linewidth=4)

        # plot rbfs
        plt.plot(rbf.centers, np.zeros(rbf.number_centers), 'gs')

        for c in rbf.centers:
            # RF prediction lines
            cx = np.arange(c - 0.7, c + 0.7, 0.01)
            cy = [rbf.basis_function(np.array([cx_]), np.array([c])) for cx_ in cx]
            plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

        plt.xlim(self.from_num, self.to_num)
        plt.show()


if __name__ == '__main__':
    fit = Fit(number_data=1000, input_dimension=1, output_dimension=1, number_centers=3, from_num=-4, to_num=4,
              data_function=Sin().implement)
    fit.fit()

    fit = Fit(number_data=1000, input_dimension=1, output_dimension=1, number_centers=10, from_num=-4, to_num=4,
              data_function=XSin().implement)
    fit.fit()
