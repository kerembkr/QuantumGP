import numpy as np
import matplotlib.pyplot as plt

def oscillatory_increasing_amplitude(x):
    # Function 1: Oscillatory function with increasing amplitude
    return x * np.sin(x)


def high_frequency_sine(x):
    # Function 2: High frequency sine wave
    return np.sin(4 * x) + 1.0


def cubic_quadratic_polynomial(x):
    # Function 3: Polynomial function with cubic and quadratic terms
    return x ** 2 + x ** 3 - 5.0


def absolute_value(x):
    # Function 4: Absolute value function (non-differentiable at x=0)
    return np.abs(x)


def exponential_growth(x):
    # Function 5: Exponential growth function
    return np.exp(x)


def sine_plus_linear(x):
    # Function 6: Combination of sine wave and linear function
    return np.sin(5 * x) + x

def sin_tanh(x):
    # Function 7: Combination of sin and tanh
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))


if __name__ == "__main__":
    x = np.linspace(-2.0, 2.0, 400)
    plt.figure()
    plt.plot(x, sin_tanh(x))
    plt.show()
