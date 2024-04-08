import numpy as np
import matplotlib.pyplot as plt

def read_matrices(filename: str,delimiter: str = '\t'):
    '''
    Reading matrices from a file

    # Parameters
    - filename: The name of the file from which the matrices are to be read
    # Returns
    - The list of matrices read from the file seperated from "#"
    '''
    matrices = []
    current_matrix = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  
            if not line or line.startswith("#"):
                if current_matrix: 
                    matrices.append(current_matrix)
                    current_matrix = []  
                continue
            
            try:
                row = [float(num) for num in line.split(delimiter)]
                current_matrix.append(row)
            except ValueError:
                # print("Skipping non-numeric line:", line)
                pass
        if current_matrix:
            matrices.append(current_matrix)
    return matrices



def linear_fit(xlist: list,ylist: list,elist: list):
    '''
    # Linear Regression
    This function finds the best fit line for a given set of data points
    Finds the fit for the equation y = a + bx
    ## Parameters
    - xlist: The x-coordinates of the data points
    - ylist: The y-coordinates of the data points
    - elist: The error in the y-coordinates of the data points. If elist=False, the function will assume that the error is 1 for all data points
    ## Returns
    - slope: The slope of the best fit line
    - intercept: The y-intercept of the best fit line
    - chi_sq: The chi-squared value of the best fit line
    '''
    # Raise an error if the lengths of xlist, ylist, and elist are not the same
    if len(xlist) != len(ylist):
        raise ValueError('The length of xlist, ylist, and elist must be the same')
    
    # If elist is False, assume that the error is 1 for all data points
    if elist == False:
        elist = [1]*len(xlist)
    # Convert the lists to numpy arrays
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    elist = np.array(elist)
    n=len(xlist)
    # Calculate the sums
    S=np.sum(1/((elist)**2))
    Sx = np.sum(xlist/((elist)**2))
    Sy = np.sum(ylist/((elist)**2))
    Sxx = np.sum((xlist**2)/((elist)**2))
    Syy = np.sum((ylist**2)/((elist)**2))
    Sxy = np.sum((xlist*ylist)/((elist)**2))

    # Calculate the slope and intercept
    Delta = S*Sxx - Sx**2

    intercept=(Sxx*Sy-Sx*Sxy)/Delta
    slope=(S*Sxy-Sx*Sy)/Delta
    # Calculate the error in the slope and intercept
    # error_intercept = np.sqrt(Sxx/Delta)
    # error_slope = np.sqrt(S/Delta)
    # cov = -Sx/Delta
    # Pearsen's correlation coefficient
    r_sq = Sxy/np.sqrt(Sxx*Syy) 

    return slope,intercept,np.sqrt(r_sq)





def polynomial_fit(xlist: list,ylist: list,sigma_list: list,degree: int,tol=1e-6):
    '''
    # Polynomial Fitting
    This function finds the best fit polynomial for a given set of data points
    Finds the fit for the equation y = a0 + a1*x + a2*x^2 + ... + an*x^n
    ## Parameters
    - xlist: The x-coordinates of the data points
    - ylist: The y-coordinates of the data points
    - sigma_list: The error in the y-coordinates of the data points
    - degree: The degree of the polynomial to be fit
    ## Returns
    - a: The coefficients of the best fit polynomial
    '''
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    sigma_list = np.array(sigma_list)
    A_matrix = np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        for j in range(degree+1):
            A_matrix[i][j] = np.sum((xlist**(i+j))/(sigma_list**2))
    B_matrix = np.zeros(degree+1)
    for i in range(degree+1):
        B_matrix[i] = np.sum((ylist*(xlist**i))/(sigma_list**2))
    # a = Gauss_seidel_solve(A_matrix.tolist(),B_matrix.tolist(),T=tol)
    a = np.linalg.solve(A_matrix,B_matrix)    
    return a,A_matrix


def modified_chebyshev_polynomial(x,degree):
    def chebyshev_polynomial(x,degree):
        if degree == 0:
            return 1
        elif degree == 1:
            return x
        else:
            return 2*x*chebyshev_polynomial(x,degree-1) - chebyshev_polynomial(x,degree-2)
    return chebyshev_polynomial(2*x - 1,degree)



# Modified Polynomial fit with the chebyshev polynomial Definitiion
def polynomial_fit_mod_chebyshev(xlist: list, ylist: list,sigma_list: list,degree: int):
    # Defining the modified chebyshev polynomial
    def modified_chebyshev_polynomial(x,degree):
        def chebyshev_polynomial(x,degree):
            if degree == 0:
                return 1
            elif degree == 1:
                return x
            else:
                return 2*x*chebyshev_polynomial(x,degree-1) - chebyshev_polynomial(x,degree-2)
        return chebyshev_polynomial(2*x - 1,degree)
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    sigma_list = np.array(sigma_list)
    A_matrix = np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        for j in range(degree+1):
            # Replace the polynomial with the modified chebyshev polynomial
            A_matrix[i][j] = np.sum((modified_chebyshev_polynomial(xlist,i)*modified_chebyshev_polynomial(xlist,j))/(sigma_list**2))
    B_matrix = np.zeros(degree+1)
    for i in range(degree+1):
        B_matrix[i] = np.sum((ylist*(modified_chebyshev_polynomial(xlist,i)))/(sigma_list**2))
    a = np.linalg.solve(A_matrix,B_matrix)    
    return a,A_matrix