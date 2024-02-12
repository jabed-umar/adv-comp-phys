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
            
    def bisection(self):
        """This method finds the root of a function using the bisection method.

        Raises:
            ValueError: f(a) and f(b) must have opposite signs
            ValueError: convergence failed

        Returns:
            float: the root of the function
        """
        a = self.a
        b = self.b
        tol = self.tol
        max_iter = self.max_iter
        
        if self.f(a)*self.f(b) > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        # check that the bounds are in the correct order
        if b<a:
            a, b = b, a   
        for i in range(max_iter):
            c = (a + b)/2
            if self.f(c) == 0 or (b - a)/2 < tol:
                return c
            if self.f(a)*self.f(c) < 0:
                b = c
            else:
                a = c
        raise ValueError("Bisection method failed after %d iterations" % max_iter)
    
    def regula_falsi(self):
        "Regula falsi method"
        a = self.a
        b = self.b
        tol = self.tol
        max_iter = self.max_iter
        if self.f(a)*self.f(b) > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        for i in range(max_iter):
            c = (a*self.f(b) - b*self.f(a))/(self.f(b) - self.f(a))
            if self.f(c) == 0 or (b - a)/2 < tol:
                return c
            if self.f(a)*self.f(c) < 0:
                b = c
            else:
                a = c
        raise ValueError("Regula falsi method failed after %d iterations" % max_iter)
    

    def newton(self, df, x0):
        """This method finds the root of a function using the Newton-Raphson method.

        Args:
            df (class): derivative of the function
            x0 (float): initial guess

        Raises:
            ValueError: if df is zero of convergence failed

        Returns:
            float: root of the function
        """
        x = x0
        for i in range(self.max_iter):
            x = x - self.f(x)/df(x)
            if abs(self.f(x)) < self.tol:
                return x
        raise ValueError("Failed to converge after %d iterations" % self.max_iter)
    
    def secant(self, x0, x1):
        """This method finds the root of a function using the secant method.

        Args:
            x0 (float): initial guess
            x1 (float): initial guess

        Raises:
            ValueError: convergence failed

        Returns:
            float: root of the function
        """
        x = x1
        x_prev = x0
        for i in range(self.max_iter):
            x, x_prev = x - self.f(x)*(x - x_prev)/(self.f(x) - self.f(x_prev)), x
            if abs(x - x_prev) < self.tol:
                return x
        raise ValueError("Secant method failed after %d iterations" % self.max_iter)
    

    def fixed_point(self, g, x0):
        """This method finds the root of a function using the fixed-point method.

        Args:
            g (class): function such that g(x) = x
            x0 (float): initial guess

        Raises:
            ValueError: convergence failed

        Returns:
            float: root of the function
        """
        x = x0
        for i in range(self.max_iter):
            x = g(x)
            if abs(x - g(x)) < self.tol:
                return x
        raise ValueError("Fixed-point method failed after %d iterations" % self.max_iter)
    