from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
''' 
    Approximation of the function f(x) using the extended trapezoid method, distributed on more processes.
    It is required to send the parameter of intervals to number of intervals + 1 because we require the rank 0 process
    to distribute the data and intervals to other processes using point to point blocking communication.
    

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : integer
        Number of sub intervals of [a,b]
    x : Array 
        Return evenly spaced numbers over a specified interval.  
        numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
        N+1 points make N sub intervals
    arrayX : Array
        Converts array x into a dtype=np.float64.
    dx : float
        Lambda, calculates the required length of each array.
    k : Array
        Used to temporary store received date from other processes.
    T : Array
        Approximation of the function using sums of each result from the other processes.
    y : Array
        Inputs f into array y.
    a : Array
        Temporary array used to store intervals and their data.
    y_left : Array
        Left endpoints.
    y_right: Array
        Right endpoints.
    G : float
        Function used to calculate the approximation using the trapezoid method.
    Returns
    -------
    Array
        Approximation of the function of f(x) from a to b using the
        trapezoid rule with N sub intervals of equal length.
'''
a = 0
b = 6
N = size-1
f = np.square
x = np.linspace(a, b, N+1)
arrayX = np.array(x, dtype=np.float64)
dx = (b - a)/N


if rank == 0:
    print(f(arrayX))
    k = np.zeros(1, dtype=np.float64)
    T = np.zeros(1, dtype=np.float64)
    for i in range(0, size-1):
        y = f(arrayX[i:i+2])
        print(f(arrayX[i:i+2]))
        comm.Send(y, dest=i+1)
        comm.Recv(k, source=i+1)
        T = T + k
    print(T)


elif rank != 0:
    a = np.zeros(2, dtype=np.float64)
    comm.Recv(a, source=0)
    y_right = a[0:1]  # right endpoint
    y_left = a[1:2]  # left endpoint
    G = (dx / 2) * np.sum(y_right + y_left)
    comm.Send(G, dest=0)
else:
    exit()

