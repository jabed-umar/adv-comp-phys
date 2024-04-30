import numpy as np
import matplotlib.pyplot as plt

class RootFindings:
    def __init__(self, f, a, b, tol, max_iter):
        """This class contains the following root-finding methods: bisection, newton, secant, fixed-point, and regula falsi.

        Args:
            f (class): function to find the root of
            a (float): initial point
            b (float): final point
            tol (float): tolerance
            max_iter (int): the maximum number of iterations
        """
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter

    #Bracket function to find the interval
    def bracket(self,t,d):
        """Brackets the root of a function.

        Args:
            a (float): lower bound of the interval.
            b (float): upper bound of the interval.
            func (class): function for which we want to find the root.
            t ( int): maximum number of iterations.
            d (float): shifting parameter.
        Returns:
        Interval : [a,b] where the root lies.
        """
        x = self.f(self.a)
        y = self.f(self.b)
        if t == 10: 
            return 0
        if x*y < 0:
            print("a=",self.a,",b=",self.b,"\nIterations:",t,"\n")
            return self.a,self.b
        t+=1
        if x*y > 0:
            if abs(x) < abs(y):  
                self.a,self.b = float(self.a-d*(self.b-self.a)),self.b
                return self.bracket(t,d)
            elif abs(x) > abs(y):
                self.a,self.b = self.a,float(self.b+d*(self.b-self.a))
                return self.bracket(t,d)

#a function to differentiate a function
def diff(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)
#### Fixed point method to solve equations ========================================================================================
def fixed_point_iteration(phi, x0, max_it=100, tolerance=1e-4):
    """
    Fixed-point iteration method for solving equations of the form x = phi(x).

    Parameters:
    - phi (function): Function defining the fixed-point iteration.
    - x0 (float): Initial guess.
    - max_it (int, optional): Maximum number of iterations. Defaults to 100.
    - tolerance (float, optional): Convergence tolerance. Defaults to 1e-4.

    Returns:
    - float: Approximate solution of the equation.

    Raises:
    - AssertionError: If the derivative of the function at the fixed point is greater than or equal to 1.
    """

    "The derivative of the function at the fixed point is greater than or equal to 1. The fixed-point iteration may not converge."
    assert abs(diff(phi,x0,10**(-5))) < 1

    for i in range(1, max_it + 1):
        x = phi(x0)
        # print(f"Iteration {i}: x = {x}")
        if abs(x - x0) < tolerance:
            return x
        x0 = x

    return x0
    


## Power iteration method to compute the eigen ========================================================================================
def Power_iteration(A, num_iterations):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    for _ in range(num_iterations):
        x = A @ x
        x = x / np.linalg.norm(x)

    eigenvalue = np.dot(x, A @ x)
    eigenvector = x

    return eigenvalue, eigenvector