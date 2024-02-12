import numpy as np
import matplotlib.pyplot as plt

class NumericalIntegration:
    def __init__(self, f, a, b, n):
        "Initialize the function f, the interval [a, b], and the number of subintervals n."
        self.a = a
        self.b = b
        self.n = n
        self.f = f

    def midpoint(self):
        "Midpoint rule for integrating f from a to b using n subintervals"
        h = (self.b - self.a) / self.n #width of each subinterval
        s = 0         #initialize sum
        for i in range(self.n):
            s += self.f(self.a + h*(0.5 + i))    # f(xn) = f(a + h*(n + 0.5))
        #print ("For n =", n)
        return s * h

    def trapezoidal(self):
        "The trapezoidal rule for integrating f from a to b using n subintervals."
        h = (self.b - self.a) / self.n # width of each subinterval
        s = 0.5*(self.f(self.a) + self.f(self.b)) # area of first and last subinterval
        for i in range(1, self.n):
            s += self.f(self.a + i*h)   # area of middle subintervals
        #print ("For n =", n)
        return s*h
    
    def simpson(self):
        "The Simpson's rule for integrating f from a to b using n subintervals."
        h = (self.b-self.a)/self.n    #width of each subinterval
        s = self.f(self.a) + self.f(self.b) #area of first and last subintervals
        if self.n % 2 == 0:
            for i in range(1, self.n, 2):
                s += 4*self.f(self.a + h*i) # add the odd terms
            for i in range(2, self.n-1, 2):
                s += 2*self.f(self.a + h*i) # add the even terms
        else:
            raise ValueError("n must be even")
        #print ("For n =", n)
        return s*h/3
    
    def legendre(self,x,n):
        "The Legendre polynomial of degree n."
        if n < 0 or type(n) != int:
            raise ValueError("n must be a non negative integer")
        if np.any(x < -1) or np.any(x > 1):
            raise ValueError("x must be in the range [-1, 1]")
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return ((2*n - 1)*x*self.legendre(x,n-1) - (n-1)*self.legendre(x, n-2))/n
        
    def plot_legendre(self,n):
        "Plot the Legendre polynomial of degree n."
        x = np.linspace(-1, 1, 1000)
        plt.plot(x, self.legendre(x, n), label=f'n = {n}')
        plt.title(f'Legendre polynomials of degree n = {n}')
        plt.xlabel('x')
        plt.ylabel(f'$P_{n}(x)$')
        plt.legend()
        plt.show()

    def legendre_derivative(self,x,n):
        "The derivative of the Legendre polynomial of degree n."
        if n == 0:
            return 0
        if x == 1 or x == -1:
            raise ValueError("x cannot be 1 or -1")
        else:
            return n * (x * self.legendre(x, n) - self.legendre(x, n-1)) / (x**2 - 1)
        
    def newton_raphson(self,f, df, x0, tol=1e-6, max_iter=100):
        "The Newton-Raphson method for solving f(x) = 0 with derivative f'(x)."
        x = x0
        # iterate until convergence 
        for i in range(max_iter):
            x_new = x - f(x)/df(x)
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        if abs(f(x)) > tol:
            raise ValueError("Failed to converge after %d iterations" % max_iter)
        return None
    
    def find_legendre_roots(self,n):
        "Find the roots of the Legendre polynomial of degree n using Newton-Raphson method."
        roots = []
        for i in range(n):
            x0 = np.cos((2*i + 1) * np.pi / (2 * n))  # Initial guess using Chebyshev nodes
            # print(x0)
            root = self.newton_raphson(lambda x: self.legendre(x, n), lambda x: self.legendre_derivative(x, n), x0)
            if root is not None:
                roots.append(root)
        return roots
    
    def gaussian_quadrature(self):
        "The Gaussian quadrature for integrating f from a to b using n subintervals."
        if self.n < 1 or type(self.n) != int:
            raise ValueError("n must be a positive integer")
        roots = self.find_legendre_roots(self.n)
        #calculate the weights
        weights = [2 / ((1 - x**2) * self.legendre_derivative(x, self.n)**2) for x in roots]   
        integral = 0
        for i in range(self.n):
            integral += weights[i] * self.f(((self.b - self.a) * roots[i])/2 + (self.b + self.a) / 2)
        return (self.b - self.a) / 2 * integral
