import numpy as np
import matplotlib.pyplot as plt
###=================================================================================== Cubic Lattice with one orbitals per site
class Crystal:
    def __init__(self, Nx, Ny, Nz):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.crr = self.create_crr()
        self.hamil = self.construct_hamil()

    def create_crr(self):
        index = np.arange(1, self.Nx * self.Ny * self.Nz + 1).reshape(self.Nx, self.Ny, self.Nz)
        return index

    def construct_hamil(self):
        NxNyNz = self.Nx * self.Ny * self.Nz
        hamil = np.zeros((NxNyNz, NxNyNz))

        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    index = self.crr[i, j, k] - 1
                    hamil[index, self.crr[(i+1) % self.Nx, j, k] - 1] = 1
                    hamil[index, self.crr[(i-1) % self.Nx, j, k] - 1] = 1
                    hamil[index, self.crr[i, (j+1) % self.Ny, k] - 1] = 1
                    hamil[index, self.crr[i, (j-1) % self.Ny, k] - 1] = 1
                    hamil[index, self.crr[i, j, (k+1) % self.Nz] - 1] = 1
                    hamil[index, self.crr[i, j, (k-1) % self.Nz] - 1] = 1

        np.fill_diagonal(hamil, 0)
        return hamil