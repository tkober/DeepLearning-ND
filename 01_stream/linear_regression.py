from numpy import *


def compute_error(b, m, points):
    totalError = 0
    # error = 1/N * sum((y - (mx+b))^2)
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + b)) ** 2
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, number_of_iterations):
    b = starting_b
    m = starting_m

    for i in range(number_of_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    # Starting points for gradients
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # Directions with respect to b and m
        # Partial derivatives for error function
        #  d/db = 2/N * sum(-(y - (mx+b)))
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        #  d/dm = 2/N * sum(-x * (y - (mx+b)))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def run():

    # Collect data
    points = genfromtxt('data.csv', delimiter=',')

    # Define hyperparameters
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initia_m = 0

    number_of_iterations = 1000

    # Train model
    print('start gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initia_m, compute_error(initial_b, initia_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initia_m, learning_rate, number_of_iterations)
    print('end point at b = {0}, m = {1}, error = {2}'.format(b, m, compute_error(b, m, points)))


if __name__ == '__main__':
    run()
