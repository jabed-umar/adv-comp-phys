import numpy as np
import matplotlib.pyplot as plt


# class Lin_alg_direct:
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
    def Gauss_Jordan(self):
        """This method solves a system of linear equations using the Gauss Jordan method

        Returns:
            list: Solution vector x
        """
        # finding the number of rows/columns of the matrix A
        n = len(self.A)
        if n > 15:
            print("The matrix is too large for this method")
            return 0
        # finding the maximum element in each column
        for i in range(n):
            max_val = self.A[i][i]
            max_row = i
            for j in range(i+1, n):
                if abs(self.A[j][i]) > abs(max_val):
                    max_val = self.A[j][i]
                    max_row = j
            # swapping the row with the maximum element with the current row
            for k in range(n):
                self.A[i][k], self.A[max_row][k] = self.A[max_row][k], self.A[i][k]
            self.b[i], self.b[max_row] = self.b[max_row], self.b[i] #=============="Here b is a list"==========
            pivot = self.A[i][i]
            # dividing the current row by the maximum element
            for l in range(n):
                self.A[i][l] = self.A[i][l] / pivot
            self.b[i] = self.b[i] / pivot
            # subtracting the current row from the other rows
            for m in range(n):
                if m != i:
                    coeff = self.A[m][i]
                    for o in range(n):
                        self.A[m][o] = self.A[m][o] - coeff * self.A[i][o]
                    self.b[m] = self.b[m] - coeff * self.b[i]
        return self.b


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
            self.b[i] = (self.b[i] - sum)/self.A[i][i]               #=============="Here b is a list"==========
        return self.b

    
    def Symmetric(self,matrix):
        """
    Check if a matrix is symmetric.

    Args:
    matrix (list of lists): The input matrix.

    Returns:
    bool: True if the matrix is symmetric, False otherwise.
    """
        # Check if the matrix is square
        if len(matrix) != len(matrix[0]):
            raise ValueError("The matrix is not square")

        # Check if the matrix is symmetric
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True
    def Decompose(self, A):
        """Cholesky Decomposition"""
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                if j == i:
                    sum = 0
                    for k in range(i):
                        sum += A[k][i] ** 2
                    A[j][j] = round((A[j][j] - sum) ** (0.5), 4)
                else:
                    A[j][i] = 0
                    sum = 0
                    for k in range(i):
                        sum += A[k][i] * A[k][j]
                    A[i][j] = round((A[i][j] - sum) / A[i][i], 4)
        return A
    
    # while running------------"""don't call Decompose and Cholesky together, it will cause an error"""----------------

    def Cholesky(self):
        """Solve linear equations using Cholesky Decomposition"""
        A = self.A
        b = self.b

        if not self.Symmetric(A):
            print("Matrix is not symmetric")
            return None

        A = self.Decompose(A)
        n = len(A)
        
        # Forward Substitution
        for i in range(n):
            sum = 0
            for k in range(i):
                sum += A[k][i] * b[k][0]
            b[i][0] = round((b[i][0] - sum) / A[i][i], 4)

        # Backward Substitution
        for i in range(n - 1, -1, -1):
            sum = 0
            for k in range(i + 1, n):
                sum += b[k][0] * A[i][k]
            b[i][0] = round((b[i][0] - sum) / A[i][i], 4)

        return b

class LinearEquationIndirect:
    def __init__(self, A, b):
        """This class contains the indirect methods (Jacobi and Gauss Seidel) to solve a system of linear equations:
        Args:
            A (Array): Square matrix of order n (n>=2) made up of coefficinets of the variables
            b (list): Vector of order n made up of constants
        """
        self.A = A
        self.b = b
        self.n = len(A)
        self.x = np.zeros(self.n)

    def diag_dmnt(self,A):
        """_summary_ : Pivoting function for producing diagonally dominant matrix by swaping columns

        Args:
            A (2-d array): sqaure matrix of order n (n>=2)

        Returns:
            2-d array : Diagonally dominant matrix
        """
        c = 0
        t = 0
        for i in range(len(A)):   #Row pivot
            t = i                 # t stores largest element of a column
            c = abs(A[i][i])      # taking absolute value of diagonal elements 
            for j in range(len(A[0])):
                if abs(A[i][j]) > c:   # checking if the element is greater than the diagonal element
                    c = abs(A[i][j])
                    t = j              
            if t > i:
                for k in range(len(A)):
                    A[k][i],A[k][t]= A[k][t],A[k][i]   
            elif t < i:
                print("Matrix is not diagonally dominant \n")
                return 0  
        return A

    def Jacobi(self, A,b,e):
        """_summary_ : Solve a system of linear equation using Jacobi method

        Args:
            A (2-array):  Square matrix of order n (n>=2) made up of coefficinets of the variables
            b (1-array): Vector of order n made up of constants
            e (precision): convergence criteria [10**(-e)]
        Returns:
            1_d array: Solution vector x
        """
        A = self.diag_dmnt(A)
        if A == 0:  #From the diag_dmnt function
            print("Jacobi not possible")
            return 0 
        n = len(A)
        C = [[1] for y in range(n)]       # C stores values after new iteration
        D = [[0] for y in range(n)]        # D stores the values after last iteration
        m = 1000                       # m stores maximum number of iterations
        sum = 0
        y = 1
        for k in range(m):
            for i in range(n):
                for j in range(n):
                    if j != i:
                        sum = sum+A[i][j]*C[j][0]
                    if abs(D[j][0]-C[j][0]) > (10**(-e)):y=1  #Checking for precision        
                if y==1:    
                    D[i][0]=(b[i][0]-sum)/A[i][i]
                else:
                    break
                sum=0  
            y = 0    
            C,D = D,C 
        print("Number of iterations is:",k+1,"\nThe solution matrix x is:\n")
        print(C)   

    ## Guass Seidel function to solve linear equations
    def Gauss_Seidel(self, A,b,e):
        """_summary_ : Solve a system of linear equation using Gauss Seidal method

        Args:
            A (2-array):  Square matrix of order n (n>=2) made up of coefficinets of the variables
            b (1-array): Vector of order n made up of constants
            e (precision): convergence criteria (10**(-e))

        Returns:
        1_d array: Solution vector x
        """
        n = len(A)
        A = self.diag_dmnt(A)
        if A == 0: # From pivot function checking the diagoanl dominance
            print("Guass-Seidal not possible")
            return 0
        x = [[0] for y in range(n)]       # x stores values after new iteration
        m = 300
        y,sum = 0,0   
        for v in range(m):
            y = 0
            for i in range(n):
                sum = 0
                for j in range(n): 
                    if j!= i:
                        sum += A[i][j]*x[j][0] 
                c=(b[i][0]-sum)/A[i][i]
                if abs(c-x[i][0]) < (10**(-e)):    #Precision condition
                    y += 1 
                x[i][0] = c 
            if y == n:   # If all elements of x follow precision condition
                break  
        print("Number of iterations is:", v+1,"\n The solution vector x for gauss seidel method is:\n") 
        print(x)
    
### Conjugate Gradient Method =====================================================================================
def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=None):
    """
    Solve the linear system Ax = b using the conjugate gradient method.

    Parameters:
    A : numpy.ndarray
        Coefficient matrix of shape (n, n).
    b : numpy.ndarray
        Right-hand side vector of shape (n,).
    x0 : numpy.ndarray
        Initial guess for the solution vector of shape (n,).
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is None (no limit).

    Returns:
    x : numpy.ndarray
        Solution vector.
    iter_count : int
        Number of iterations performed.
    """

    n = len(b)
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    iter_count = 0

    while True:
        iter_count += 1
        Ap = np.dot(A, p)
        alpha = np.dot(r.T, r) / np.dot(p.T, Ap)
        x += alpha * p
        r_next = r - alpha * Ap
        if np.linalg.norm(r_next) < tol:
            break
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r)
        p = r_next + beta * p
        r = r_next

        if max_iter is not None and iter_count >= max_iter:
            break

    return x, iter_count

## Conjugate Gradient Method without storing the matrix ===========================================================
def dot_product(A, b):
    # here A is a function and b is a vector
    n = len(b)
    result = []
    for i in range(n):
        dot_product_sum = 0
        for j in range(n):
            dot_product_sum += A(i, j) * b[j]
        result.append(dot_product_sum)
    return result

def conjugate_gradient_no_store(A, b, x0, tol = 1e-4, max_iter = 10):
    """This function solves the linear system Ax = b using the conjugate gradient method without storing the residuals

    Args:
        A (array): Coefficient matrix of shape (n, n).
        b (array): Right-hand side vector of shape (n,).
        x0 (array): Initial guess for the solution vector of shape (n,).
        tol (float, optional): tolerance . Defaults to 1e-4.
        max_iter (int, optional): maximum no of iteration. Defaults to 10.

    Returns:
        array: Solution vector, no of iterations, residuals
    """
    n = len(b)
    x = x0.copy()
    r = b - dot_product(A, x)
    p = r.copy()
    iter_count = 0

    residues = []
    while True:
        iter_count += 1
        Ap = dot_product(A, p)
        alpha = np.dot(r.T, r) / np.dot(p.T, Ap)
        x += alpha * p
        r_next = r - alpha * Ap
        residues.append(np.linalg.norm(r_next))
        if np.linalg.norm(r_next) < tol:
            break
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r)
        p = r_next + beta * p
        r = r_next

        if max_iter is not None and iter_count >= max_iter:
            break

    return x, iter_count, residues



## Power iteration method to compute the eigen ========================================================================================
def power_iteration(matrix, tolerance=1e-10, max_iterations=1000):
    """This code finds out the maximum eigenvalues and eigenvectors of a matrix

    Args:
        matrix (array): given matrix
        tolerance (float, optional): the tolerance. Defaults to 1e-10.
        max_iterations (int, optional): the maximum no of iterations. Defaults to 1000.

    Returns:
        eigvalue, eigvector: float and list
    """
    # Initialize a random vector of appropriate size
    n = matrix.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize the initial vector
    
    # Power iteration loop
    for _ in range(max_iterations):
        x_new = np.dot(matrix, x)  # Multiply matrix with the vector
        eigenvalue = np.dot(x, x_new)  # Approximate the eigenvalue
        x_new = x_new / np.linalg.norm(x_new)  # Normalize the new vector
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
    
    eigenvector = x_new
    return eigenvalue, eigenvector


##### QR Decomposition method ===============================================================================================================
def QR_factorize(A):
    """This function factorise the matrix A into Q and R

    Args:
        A (ndarray): The given array

    Returns:
        ndarray: Q and R matrices 
    """
    A = np.array(A) if type(A) != np.ndarray else A
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    for i in range(A.shape[1]):
        u_i = A[:,i]
        sum = 0
        for j in range(i):
            sum += np.dot(A[:,i],Q[:,j])*Q[:,j]
        u_i = u_i - sum
        Q[:,i] = u_i/np.linalg.norm(u_i)
        for j in range(i+1):
            R[j,i] = np.dot(A[:,i],Q[:,j])
            
    return Q,R


def eigen_val_QR(A,tolerance = 1e-6):
    """This find eigenvalues of given matrix A 

    Args:
        A (ndarray): The given matrix A
        tolerance (float, optional): The tolerance to determine the eigenvalues. Defaults to 1e-6.

    Returns:
        ndarray: eigenvales of matrix A 
    """
    A = np.array(A)
    copy_A = np.copy(A)
    Q,R = QR_factorize(A)
    A = np.matmul(R,Q)
    i=1
    while np.linalg.norm(A-copy_A)>tolerance:
        copy_A = np.copy(A)
        Q,R = QR_factorize(A)
        A = np.matmul(R,Q)
        i+=1
    return np.diag(A),i