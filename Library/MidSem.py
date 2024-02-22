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
    

#--------------------------------------------------RK for shooting method--------------------------------------
def rk_shoot(d2ydx2, dydx, x0, y0, z0, xf, h):
    """shooting method using RK4.

    Args:
        d2ydx2 (array): derivative of the second order.
        dydx (array): derivative of the first order.
        x0 (float): initial value of x.
        y0 (float): initial value of y.
        z0 (float): initial value of z.
        xf (float): final value of x.
        h (float): strip size.

    Returns:
        array: solution of the differential equation.
    """
    x = [x0]
    y = [y0]
    z = [z0]
    # no of strips
    N = int((xf-x0)/h)
    for i in range(N):
        k1 = h * dydx(x[i], y[i], z[i])
        l1 = h * d2ydx2(x[i], y[i], z[i])

        k2 = h * dydx(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        l2 = h * d2ydx2(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)

        k3 = h * dydx(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        l3 = h * d2ydx2(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)

        k4 = h * dydx(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * d2ydx2(x[i] + h, y[i] + k3, z[i] + l3)

        x.append(x[i] + h)
        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        z.append(z[i] + (l1 + 2*l2 + 2*l3 + l4)/6)
    return x, y, z

# Lagrange interpolation for the intrpolation part 
def lag_inter(zeta_h, zeta_l, yh, yl, y):
    zeta = zeta_l + (zeta_h - zeta_l) * (y - yl)/(yh - yl)
    return zeta

# Shooting method for solving the 2nd order ODE
def shoot(d2ydx2, dydx, x0, y0, xf, yf, z1, z2, h, tol=1e-6):  
    x, y, z = rk_shoot(d2ydx2, dydx, x0, y0, z1, xf, h)  #use RK4 
    yn = y[-1]
    if abs(yn - yf) > tol:
        if yn < yf:
            zeta_l = z1
            yl = yn
            x, y, z = rk_shoot(d2ydx2, dydx, x0, y0, z2, xf, h)
            yn = y[-1]
            if yn > yf:
                zeta_h = z2
                yh = yn
                zeta = lag_inter(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = rk_shoot(
                    d2ydx2, dydx, x0, y0, zeta, xf, h)
                return x, y
            else:
                print("Invalid bracketing.")
        elif yn > yf:
            zeta_h = z1
            yh = yn
            x, y, z = rk_shoot(d2ydx2, dydx, x0, y0, z2, xf, h)
            yn = y[-1]
            if yn < yf:
                zeta_l = z2
                yl = yn
                zeta = lag_inter(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = rk_shoot(
                    d2ydx2, dydx, x0, y0, zeta, xf, h)
                return x, y
            else:
                print("Invalid bracketing.")
    else:
        return x, y
    

class HeatEquationSolver:
    def __init__(self, temp0: callable, L: float, T: float, nL: int, nT: int, t_upto: int = None):
        """Solves the heat equation using the explicit and implicit methods.

        Args:
            temp0 (callable): initial temperature distribution.
            L (float): length of the rod.
            T (float): time period.
            nL (int): no. of strips in the length.
            nT (int): no. of strips in the time.
            t_upto (int): time upto which the solution is to be plotted. default is nT.

        Returns:
            2D list: solution of the heat equation.
        """
        self.temp0 = temp0
        self.L = L
        self.T = T
        self.nL = nL
        self.nT = nT
        self.t_upto = t_upto if t_upto is not None else nT
    
    def explicit_solve(self):
        """This function solves the heat equation using the explicit method."""
        ht = self.T / self.nT
        hx = self.L / self.nL
        alpha = ht / (hx ** 2) #""alpha, the stability factor, must be less than 0.5 for the explicit method""
        print("The stability factor (alpha) is", alpha)
        if alpha > 0.5:
            raise ValueError("alpha must be less than 0.5")
        A = [[0 for _ in range(self.nL)] for _ in range(self.t_upto)]
        for i in range(self.nL):
            A[0][i] = self.temp0(i, self.nL)

        for t in range(1, self.t_upto):
            for x in range(self.nL):
                if x == 0:
                    A[t][x] = A[t - 1][x] * (1 - 2 * alpha) + A[t - 1][x + 1] * alpha
                elif x == self.nL - 1:
                    A[t][x] = A[t - 1][x - 1] * alpha + A[t - 1][x] * (1 - 2 * alpha)
                else:
                    A[t][x] = A[t - 1][x - 1] * alpha + A[t - 1][x] * (1 - 2 * alpha) + A[t - 1][x + 1] * alpha
        return A
    

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
    

class LinearEquationDirect:
    def __init__(self, A, b):
        """This class contains the direct methods (Gauss_Jordan, LU and Cholesky) to solve a system of linear equations:
        Args:
            A (Array): Square matrix of order n (n>=2) made up of coefficinets of the variables
            b (list): Vector of order n made up of constants
        """
        self.A = A
        self.b = b
        self.det = 1
        self.n = len(A)
        self.inverse = None
    def LU_decompose(self):
        """_summary_ : LU decomposition method to solve linear equation Ax=b"""
        #finding the number of rows/columns
        n  = len(self.A)
        #convert the matrix to upper and lower triangular matrix
        for j in range(n):
            for i in range(n):
                if i <= j :
                        sum = 0
                        for k in range(i):
                            sum += self.A[i][k]*self.A[k][j]
                        self.A[i][j] = self.A[i][j] - sum
                else  :
                        sum = 0
                        for k in range(j):
                            sum += self.A[i][k]*self.A[k][j]
                        self.A[i][j] = (self.A[i][j] - sum)/self.A[j][j]      
    #forward substitution
        for i in range(n):
            sum = 0
            for j in range(i):
                sum += self.A[i][j]*self.b[j]
            self.b[i] = self.b[i] - sum      
    #backward substitution
        for i in range(n-1,-1,-1):
            sum = 0 
            for j in range(i+1,n):
                sum += self.A[i][j]*self.b[j]
            self.b[i] = (self.b[i] - sum)/self.A[i][i]
        return self.b