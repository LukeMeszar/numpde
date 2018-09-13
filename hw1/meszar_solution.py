import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

def diffmat(n, h):
    D = np.zeros((n,n))
    D[0,0] = -3
    D[0,1] = 4
    D[0,2] = -1
    D[n-1,-3] = 1
    D[n-1,-2] = -4
    D[n-1,-1] = 3
    for i in range(1,n-1):
        D[i,i-1] = -1
        D[i,i+1] = 1
    return (1/(2*h))*D

def test_first_derivative(f, fp, n, domain):
    x = np.linspace(domain[0], domain[1], n)
    h = (domain[1] - domain[0])/(n-1)
    u = f(x)
    D = diffmat(n,h)
    u_prime = np.matmul(D,u)
    #return x, u_prime
    plt.plot(x,u_prime, 'o')
    plt.plot(x,fp(x))
    plt.show()

def grid_refinement_error(f, fp=None, fpp=None, diff, domain):
    grids = 2**np.arange(3,10)
    error = []
    for n in grids:
        x = np.linspace(domain[0], domain[1], n)
        xx, yy = diff(f, fp, n, domain)
        error.append(np.linalg.norm(yy-fp(xx), np.inf))
    plt.loglog(grids, error, 'o')
    plt.loglog(grids, grids**(-2.), label='$1/n^2$')
    plt.loglog(grids, 12*grids**(-2.), label='$1/(12n^2)$')
    plt.show()

def func1(x):
    return x**2 + np.cos(x)

def func1p(x):
    return 2*x - np.sin(x)

#https://www.geometrictools.com/Documentation/FiniteDifferences.pdf
def diff2mat(n,h):
    D = np.zeros((n,n))
    D[0,0] = 1
    D[0,1] = -2
    D[0,2] = 1
    D[n-1,-3] = 1
    D[n-1,-2] = -2
    D[n-1,-1] = 1
    for i in range(1,n-1):
        D[i,i-1] = 1
        D[i,i] = -2
        D[i,i+1] = 1
    return (1/h**2)*D

def test_second_derivative(f, fpp, n, domain):
    x = np.linspace(domain[0], domain[1], n)
    h = (domain[1] - domain[0])/(n-1)
    u = f(x)
    D = diff2mat(n,h)
    u_pp= np.matmul(D,u)
    #return x, u_prime
    # print(f(x))
    # print(fpp(x))
    plt.plot(x,u_pp, 'o')
    plt.plot(x,fpp(x))
    plt.show()

def func2(x):
    return np.sin(x)

def func2pp(x):
    return -np.sin(x)

if __name__ == '__main__':
    #test_first_derivative(func1, func1p, 10, (-2,2))
    #test_first_derivative(np.sin, np.cos, 200, (-np.pi,np.pi))
    #grid_refinement_error(np.sin,np.cos,test_first_derivative, (-np.pi,np.pi))
    test_second_derivative(func2, func2pp, 10, (-1,1))
