from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


def trapz(f, a, b, N=50):
    '''Aproksimacija funkcije f(x) produljenom trapeznom formulom.

    The trapezoid rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/2) \sum_{k=1}^N (f(x_k) + f(x_{k-1}))
    where x_k = a + k*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using the
        trapezoid rule with N subintervals of equal length.
    '''
    x = np.linspace(a, b, N + 1)  # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:]  # right endpoints
    y_left = y[:-1]  # left endpoints
    dx = (b - a) / N
    T = (dx / 2) * np.sum(y_right + y_left)
    return T


T = trapz(np.square, 0, 6, 6)
print(T)
# crtanje grafa za prikaz funkcije
a = 0
b = 6
f = np.square
N = 6

# x and y values for the trapezoid rule
x = np.linspace(a, b, N + 1)
y = f(x)

# X and Y values for plotting y=f(x)
X = np.linspace(a, b, N + 1)
Y = f(X)
plt.plot(X, Y)

for i in range(N):
    xs = [x[i], x[i], x[i + 1], x[i + 1]]
    ys = [0, f(x[i]), f(x[i + 1]), 0]
    plt.fill(xs, ys, 'b', edgecolor='b', alpha=0.2)

plt.title('Trapezoid Rule, N = {}'.format(N))
plt.show()
