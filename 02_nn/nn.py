from numpy import exp, array, random, dot


class NeuralNetwork():

    def __init__(self):
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3,1)) - 1

    # Activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Simgmoid curve gradient
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for iteration in range(1, iterations):
            # Pass the training data throug the NN
            output = self.predict(training_set_inputs)

            # Calculate the error
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__  == '__main__':

    # Initialze a single layer feed forward neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_network.synaptic_weights)

    # Traing set. 4 examples with 3 input values and 1 output value each
    #
    #   Input           Output
    #   0   0   1       0
    #   1   1   1       1
    #   1   0   1       1
    #   0   1   1       0
    #
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the NN 10,000 times
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training:')
    print(neural_network.synaptic_weights)

    # Test the NN
    print('Result for [1, 0, 0] --> ?: ')
    print(neural_network.think(array([1, 0, 0])))
