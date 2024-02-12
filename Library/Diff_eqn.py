import numpy as np
import matplotlib.pyplot as plt

class DifferentialEquation1():
    def __init__(self, func, x0, x1, y0, h):
        """This class contains different methods for solving first order ordinary differential equations.

        Args:
            func (class): the function to solve
            y0 (float): initial condition
            x0 (float): initial point
            x1 (float): final point
            h (float): step size
        """
        self.func = func
        self.y0 = y0
        self.x0 = x0
        self.x1 = x1
        self.h = h
        self.x = np.arange(x0, x1, h)
        self.y = np.zeros(len(self.x))
        self.y[0] = y0

    def forward_euler(self):
        """This method solves the differential equation using the forward Euler method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            # y[i] = y[i-1] + h*f(x[i-1], y[i-1])
            self.y[i] = self.y[i-1] + self.h*self.func(self.x[i-1], self.y[i-1])
            y.append(self.y[i])
        plt.plot(x, y, label='Forward Euler')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.legend()
        plt.show()
        return x, y

    def backward_euler(self):
        """This method solves the differential equation using the backward Euler method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            # y[i] = y[i-1] + h*f(x[i], y[i])
            self.y[i] = (self.y[i-1] + self.h*self.func(self.x[i], self.y[i]))
            y.append(self.y[i])
        plt.plot(x, y, label='Backward Euler')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.legend()
        plt.show()
        return x, y
    
    
    def predictor_corrector(self):
        """This method solves the differential equation using the predict-corrector method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            y_pred = self.y[i-1] + self.h*self.func(self.x[i-1], self.y[i-1])
            self.y[i] = self.y[i-1] + (self.h/2)*(self.func(self.x[i-1], self.y[i-1]) + self.func(self.x[i], y_pred))
            y.append(self.y[i])
        plt.plot(x, y, label='Predict-Corrector')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.legend()
        plt.show()
        return x, y
    
    def runge_kutta2(self):
        """This method solves the differential equation using the Runge-Kutta2 method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            k1 = self.h*self.func(self.x[i-1], self.y[i-1])
            k2 = self.h*self.func(self.x[i-1] + self.h/2, self.y[i-1] + k1/2)
            self.y[i] = self.y[i-1] + k2
            y.append(self.y[i])
        plt.plot(x, y, label='Runge-Kutta 2')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.legend()
        plt.show()
        return x, y

    def runge_kutta4(self):
        """This method solves the differential equation using the Runge-Kutta method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            k1 = self.h*self.func(self.x[i-1], self.y[i-1])
            k2 = self.h*self.func(self.x[i-1] + (self.h/2), self.y[i-1] + (k1/2))
            k3 = self.h*self.func(self.x[i-1] + (self.h/2), self.y[i-1] + (k2/2))
            k4 = self.h*self.func(self.x[i-1] + self.h, self.y[i-1] + k3)
            self.y[i] = self.y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
            y.append(self.y[i])
        plt.plot(x, y,color='black')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.title(f"Solution of the diff eqn using RK4 with h = {self.h}", fontsize=13, color='b', fontweight='bold')
        plt.show()
        return x, y
    
    

class DifferentialEquation2():
    def __init__(self,d2ydx2,dydx, x0, x1, y0, z0, h): 
        """This class contains different methods for solving second order ordinary differential equations.

        Args:
            d2ydx2 (class): the second derivative of the function to solve
            dydx (class): the first derivative of the function to solve
            y0 (float): initial condition for y
            z0 (float): initial condition for z
            x0 (float): initial point 
            x1 (float): final point
            h (float): step size
        """
        self.d2ydx2 = d2ydx2
        self.dydx = dydx
        self.y0 = y0
        self.z0 = z0
        self.x0 = x0
        self.x1 = x1
        self.h = h
        self.x = np.arange(x0, x1, h)
        self.y = np.zeros(len(self.x))
        self.z = np.zeros(len(self.x))
        self.y[0] = y0
        self.z[0] = z0

    def runge_kutta4(self):
        """This method solves the differential equation using the Runge-Kutta method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.y0]
        z = [self.z0]
        for i in range(1, len(self.x)):
            k1 = self.h*self.dydx(self.x[i-1], self.y[i-1], self.z[i-1])
            l1 = self.h*self.d2ydx2(self.x[i-1], self.y[i-1], self.z[i-1])
            k2 = self.h*self.dydx(self.x[i-1] + (self.h/2), self.y[i-1] + (k1/2), self.z[i-1] + (l1/2))
            l2 = self.h*self.d2ydx2(self.x[i-1] + (self.h/2), self.y[i-1] + (k1/2), self.z[i-1] + (l1/2))
            k3 = self.h*self.dydx(self.x[i-1] + (self.h/2), self.y[i-1] + (k2/2), self.z[i-1] + (l2/2))
            l3 = self.h*self.d2ydx2(self.x[i-1] + (self.h/2), self.y[i-1] + (k2/2), self.z[i-1] + (l2/2))
            k4 = self.h*self.dydx(self.x[i-1] + self.h, self.y[i-1] + k3, self.z[i-1] + l3)
            l4 = self.h*self.d2ydx2(self.x[i-1] + self.h, self.y[i-1] + k3, self.z[i-1] + l3)
            self.y[i] = self.y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
            self.z[i] = self.z[i-1] + (l1 + 2*l2 + 2*l3 + l4)/6
            x.append(self.x[i])
            y.append(self.y[i])
            z.append(self.z[i])
        plt.plot(x, y, label='Runge-Kutta 4')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.legend()
        plt.show()
        return x, y
    


# class SympleticIntegrator:
    

class BoundaryValue:
    def __init__(self, d2ydx2, dydx, x0, x1, alpha, beta,z0 ,h):
        """This class contains different methods for solving second order ordinary differential equations.

        Args:
            d2ydx2 (class): the second derivative of the function to solve
            dydx (class): the first derivative of the function to solve
            x0 (float): initial point 
            x1 (float): final point
            alpha (float): initial condition for y
            beta (float): final condition for z
            h (float): step size
        """
        self.d2ydx2 = d2ydx2
        self.dydx = dydx
        self.x0 = x0
        self.x1 = x1
        self.alpha = alpha
        self.beta = beta
        self.z0 = z0
        self.h = h
        self.x = np.arange(x0, x1, h)
        self.y = np.zeros(len(self.x))
        self.y[0] = alpha
        self.z[0] = z0

    def shooting(self):
        """This method solves the differential equation using the shooting method.

        Returns:
            list: x and y values
        """
        x = [self.x0]
        y = [self.alpha]
        z = [self.z0]
        for i in range(1, len(self.x)):
            x.append(self.x[i])
            k1 = self.h*self.dydx(self.x[i-1], self.y[i-1], self.z[i-1])
            l1 = self.h*self.d2ydx2(self.x[i-1], self.y[i-1], self.z[i-1])
            k2 = self.h*self.dydx(self.x[i-1] + (self.h/2), self.y[i-1] + (k1/2), self.z[i-1] + (l1/2))
            l2 = self.h*self.d2ydx2(self.x[i-1] + (self.h/2), self.y[i-1] + (k1/2), self.z[i-1] + (l1/2))
            k3 = self.h*self.dydx(self.x[i-1] + (self.h/2), self.y[i-1] + (k2/2), self.z[i-1] + (l2/2))
            l3 = self.h*self.d2ydx2(self.x[i-1] + (self.h/2), self.y[i-1] + (k2/2), self.z[i-1] + (l2/2))
            k4 = self.h*self.dydx(self.x[i-1] + self.h, self.y[i-1] + k3, self.z[i-1] + l3)
            l4 = self.h*self.d2ydx2(self.x[i-1] + self.h, self.y[i-1] + k3, self.z[i-1] + l3)
            self.y[i] = self.y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
            self.z[i] = self.z[i-1] + (l1 + 2*l2 + 2*l3 + l4)/6
            y.append(self.y[i])
            z.append(self.z[i])
            return x, y, z
            

    # Lagrange interpolation for the intrpolation part 
    def lag_inter(self,zeta_h, zeta_l, yh, yl, y):
        zeta = zeta_l + (zeta_h - zeta_l) * (y - yl)/(yh - yl)
        return zeta
        
        # Shooting method for solving the 2nd order ODE
    # def shoot(self, yf, z1, z2, tol):
    #     """This method solves the differential equation using the shooting method.

    #     Args:
    #         yf (float): final condition for y
    #         z1 (float): initial guess for z
    #         z2 (float): initial guess for z
    #         tol (float): tolerance

    #     Returns:
    #         list: x and y values
    #     """
    #     while True:
    #         x, y, z1 = self.shooting()
    #         if abs(y[-1] - yf) < tol:
    #             return x, y
    #         z1, z2 = z2, self.lag_inter(y[-1], y[-2], z1, z2, yf)
    #     return x, y
    # ## needs to complete

    import numpy as np

class CrankNicolsonSolver:
    def __init__(self, g, n, T, alpha, gamma):
        """This function solves the heat equation using the Crank-Nicolson method.

    Args:
        g (class): The initial condition of the heat equation or g(x) = u(x, 0)
        n (int): the number of steps in the x-direction
        T (int): the number of steps in the t-direction
        alpha (float): delta(t)/delta(x)^2 (time step divided by space step squared)
        gamma (float): the coefficient of $frac{\partial^2 u}{\partial x^2}$

    Returns:
        array: a list of solutions to the heat equation
    """
        self.g = g
        self.n = n
        self.T = T
        self.alpha = alpha
        self.gamma = gamma

    def heat_eqn_solve(self):
        m = self.alpha * self.gamma
        # Initialize vector 'u' with values
        u = np.zeros(self.n)
        for i in range(len(u)):
            u[i] = self.g(i)

        # Create identity matrix 'I' and tridiagonal matrix 'B'
            '''
    The matrix A is a tridiagonal matrix with  `2 + m' on the diagonal and `-m' on the off-diagonals
    The matrix B is a tridiagonal matrix with `2 - m' on the diagonal and `m' on the off-diagonals
    '''
        I = np.identity(self.n)
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            B[i, i] = 2
        for j in range(self.n - 1):
            B[j, j + 1] = -1
        for j in range(1, self.n):
            B[j, j - 1] = -1

        # Construct matrices 'm1' and 'm2' for the Crank-Nicolson method
        A1 = 2 * I - m * B
        A2 = np.linalg.inv(2 * I + m * B)

        # Initialize vector 'u' with values and an empty list 'solutions' to store solutions
        u = np.array(u)
        solutions = []
        i = 0

        # Perform iterations to solve the linear system using Crank-Nicolson method
        while i < self.T:
            u = A1 @ A2 @ u
            solutions.append(u)
            i += 1   # Increment the time step by 1, i.e. delta(t) = 1

        # Return the list of solutions
        return solutions

