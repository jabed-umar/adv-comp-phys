import copy
import numpy as np
import matplotlib.pyplot as plt

#Differentiation class

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
    

def verlet(x0, v0, a, t, h):
    """This function solves the differential equation using the Verlet method.

    Args:
        x0 (float): initial position
        v0 (float): initial velocity
        a (class): acceleration
        t (float): final time
        h (float): step size

    Returns:
        list: x and v values
    """
    x = [x0]
    v = [v0]
    x.append(x0 + v0*h + 0.5*a(x0)*h**2)
    for i in range(1, int(t/h)):
        x.append(2*x[i] - x[i-1] + a(x[i])*h**2)
        v.append((x[i+1] - x[i-1])/(2*h))
    return x, v

def leapfrog(x0, v0, a, t, h):
    """This function solves the differential equation using the leapfrog method.

    Args:
        x0 (float): initial position
        v0 (float): initial velocity
        a (class): acceleration
        t (float): final time
        h (float): step size

    Returns:
        list: x and v values
    """
    x = [x0]
    v = [v0]
    v.append(v0 + a(x0)*h/2)
    for i in range(1, int(t/h)):
        x.append(x[i-1] + v[i]*h)
        v.append(v[i] + a(x[i])*h)
    return x, v

 
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
     
import numpy as np

class HEImplicitSolver:
    def __init__(self, g, a, b, x0, x_m, t0, t_m, N_x, N_t, req_time_step, iflist=True):
        '''
    # Implicit Finite Difference Method
    for solving the heat equation
    ## Parameters
    - g: Initial condition function u(x,t=0) = g(x)
    - a: Boundary condition function u(x=0,t) = a(t)
    - b: Boundary condition function u(x=x_m,t) = b(t)
    - x0: Initial value of x
    - x_m: Final value of x
    - t0: Initial value of t
    - t_m: Final value of t
    - N_x: Number of steps to divide the interval [x0,x_m]
    - N_t: Number of steps to divide the interval [t0,t_m]
    - req_time_step: The time step to which the solution is to be calculated
    - iflist: If True, the function will return the list of u values, if False, the function will return u as a column matrix or a vector
    ## Returns
    - x: List of x values
    - t: List of t values
    - u: List of List of u values or vector depending on the value of iflist
    '''
        self.g = g
        self.a = a
        self.b = b
        self.x0 = x0
        self.x_m = x_m
        self.t0 = t0
        self.t_m = t_m
        self.N_x = N_x
        self.N_t = N_t
        self.req_time_step = req_time_step
        self.iflist = iflist

    def implicit_solve(self):
        hx = (self.x_m - self.x0) / self.N_x
        ht = (self.t_m - self.t0) / self.N_t
        x = [self.x0 + i * hx for i in range(1, self.N_x)]
        alpha = ht / (hx**2)

        if alpha > 0.5:
            raise ValueError("The value of alpha should be less than 0.5")

        u = [[self.g(i)] for i in x]
        A = [[0 for _ in range(self.N_x - 1)] for _ in range(self.N_x - 1)]

        for i in range(len(A)):
            for j in range(len(A[i])):
                if i == j:
                    A[i][j] = 1 + 2 * alpha
                elif abs(i - j) == 1:
                    A[i][j] = -alpha

        A1 = np.linalg.inv(A)
        del A
        An = np.linalg.matrix_power(A1, self.req_time_step)
        del A1
        v_req = np.matmul(An, u).tolist()
        del An
        v_req.insert(0, [self.a(self.t0)])
        v_req.append([self.b(self.t0)])
        x.insert(0, self.x0)
        x.append(self.x_m)
        ulist = []

        if not self.iflist:
            return x, v_req, [self.t0 + i * ht for i in range(self.N_t + 1)]
        else:
            for i in range(len(v_req)):
                ulist.append(v_req[i][0])
            return x, ulist, [self.t0 + i * ht for i in range(self.req_time_step + 1)]



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
        #construct the tridiagonal matrix B
        for i in range(self.n):
            B[i, i] = 2
        for j in range(self.n - 1):
            B[j, j + 1] = -1
        for j in range(1, self.n):
            B[j, j - 1] = -1

        # Construct matrices 'A1' and 'A2' for the Crank-Nicolson method
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



## _____________________________________Possion Equation_______________________________________
def poission_eqn(u, func, xlim=2, ylim=1):
    u = copy.deepcopy(u.T)

    N = u.shape[0]
    h = ylim / (N - 1)
    x = np.linspace(0, xlim, N)
    y = np.linspace(0, ylim, N)

    N2 = (N-1)**2
    A=np.zeros((N2, N2))
    coo = lambda i, j, N: i * N + j
    for i in range(N-1):
        for j in range(N-1):

            this = coo(i, j, N-1)
            A[this, this] = 4  # self

            if i > 0:
                A[this, coo(i-1, j, N-1)] = -1  # Left
            if i < N-2:
                A[this, coo(i+1, j, N-1)] = -1  # Right
            if j > 0:
                A[this, coo(i, j-1, N-1)] = -1  # Up
            if j < N-2:
                A[this, coo(i, j+1, N-1)] = -1  # Down

    r = np.zeros(N2)
    # vector r      
    for i in range(N-1):
        for j in range(N-1):           
            r[i+(N-1)*j] = (h**2) * func(x[i+1], y[j+1])

    # Boundary
    b_bottom_top=np.zeros(N2)
    for i in range(0,N-1):
        b_bottom_top[i]= x[i+1] #Bottom Boundary
        b_bottom_top[i+(N-1)*(N-2)] = x[i+1] * np.e# Top Boundary

    b_left_right=np.zeros(N2)
    for j in range(0, N-1):
        b_left_right[(N-1)*j] = 0 # Left Boundary
        b_left_right[N-2+(N-1)*j] = 2*np.exp(y[j+1])# Right Boundary

    b = b_left_right + b_bottom_top

    C = np.linalg.inv(A) @ (b - r)

    u[1:N, 1:N] = C.reshape((N-1, N-1))
    return u

       
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
    x, y, z = rk_shoot(d2ydx2, dydx, x0, y0, z1, xf, h)
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
#--------------------------------------------------RK for shooting method End--------------------------------------
    

## ___________Finite element method for solving 2nd order ODE____________________
    