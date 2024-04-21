import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.sparse as sp
from scipy.sparse import coo_array

###################======================Weyl Nodal Loop Hamiltonian===============================

###====================Hamiltonian in momentum space
class HamWeyl:
    def __init__(self, t_x, t_y, m, t_2):
        self.t_x = t_x
        self.t_y = t_y
        self.m = m
        self.t_2 = t_2
    def hamiltonian(self, k_x, k_y, k_z):
        """This function returns the Hamiltonian of the 3D model.
    Args:
        k_x (float): Momentum in x direction
        k_y (float): Momentum in y direction
        k_z (float): Momentum in z direction
        t_x (float): Hopping in x direction
        t_y (float): Hopping in y direction
        m (float): Onsite energy
        t_2 (float): Hopping in z direction
    Returns:
        array: The Hamiltonian of the 3D, 2 band model
    """ 
        d_1 = self.t_x * np.cos(k_x) + self.t_y * np.cos(k_y) + np.cos(k_z) - self.m 
        d_2 = self.t_2 * np.sin(k_z)
        d_3 = d_1 - 1j * d_2
        matrix = np.array([[0, d_3], [np.conjugate(d_3), 0]])
        return matrix
    
    def eigen(self, k_x, k_y, k_z):
        """This function returns the eigenvalues and eigenstates of the Hamiltonian as a function of k_x and k_y."""
        matrix = self.hamiltonian(k_x, k_y, k_z)
        e, v = np.linalg.eigh(matrix)
        return e, v

 
    def plot_eigenvalues(self, k_z):
        """This function plots the band structure in 3D of the Hamiltonian as a function of k_x and k_y."""
        k_points = 200
        k_x_values = np.linspace(-np.pi, np.pi, k_points)
        k_y_values = np.linspace(-np.pi, np.pi, k_points)
        k_x_mesh, k_y_mesh = np.meshgrid(k_x_values, k_y_values)
        eigenvalues = np.zeros((k_points, k_points, 2))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(k_points):
            for j in range(k_points):
                k_x = k_x_mesh[i, j]
                k_y = k_y_mesh[i, j]
                matrix = self.hamiltonian(k_x, k_y, k_z)
                eigenvalues[i, j] = np.linalg.eigvalsh(matrix)

        for band in range(2):
            eigenvalues_band = eigenvalues[:, :, band]
            cmap = plt.get_cmap('cool' if band == 0 else 'hot')
            ax.plot_surface(k_x_mesh, k_y_mesh, eigenvalues_band, cmap=cmap)

        ax.set_xlabel('$k_x$', fontsize=10, labelpad=6, fontweight='bold')
        ax.set_ylabel('$k_y$', fontsize=10, labelpad=6, fontweight='bold')
        ax.set_zlabel('Energy', fontsize=10, labelpad=6, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        title = f"Parameters: $t_x$={self.t_x}, $t_y$={self.t_y}, $m$={self.m}, $t_2$={self.t_2}"
        ax.set_title(title, fontsize=14)
        ax.view_init(elev=25, azim=35)
        plt.show()

    def plot_eigen(self, k_z):
        """This function plots the band structure in 2D (heatmap) of the Hamiltonian as a function of k_x and k_y."""
        k_points = 200
        k_x_values = np.linspace(-np.pi, np.pi, k_points)
        k_y_values = np.linspace(-np.pi, np.pi, k_points)
        k_x_mesh, k_y_mesh = np.meshgrid(k_x_values, k_y_values)
        eigenvalues = np.zeros((k_points, k_points, 2))

        for i in range(k_points):
            for j in range(k_points):
                k_x = k_x_mesh[i, j]
                k_y = k_y_mesh[i, j]
                matrix = self.hamiltonian(k_x, k_y, k_z)
                eigenvalues[i, j] = np.linalg.eigvalsh(matrix)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for band in range(2):
            eigenvalues_band = eigenvalues[:, :, band]
            cmap = plt.get_cmap('hot' if band == 0 else 'viridis')
            im = axes[band].imshow(eigenvalues_band, extent=(-np.pi, np.pi, -np.pi, np.pi), cmap=cmap)
            axes[band].set_title(f'Band {band + 1}')
            axes[band].set_xlabel('$k_x$', fontsize=10, labelpad=6, fontweight='bold')
            axes[band].set_ylabel('$k_y$', fontsize=10, labelpad=6, fontweight='bold')
            fig.colorbar(im, ax=axes[band], orientation='vertical')
        fig.suptitle(f"Parameters: $t_x$={self.t_x}, $t_y$={self.t_y}, $m$={self.m}, $t_2$={self.t_2}", fontsize=16)
        plt.tight_layout()
        plt.show()

###============================================= Cubic Lattice with two orbitals per site
#===================This is helpful to study the smaller system only (don't run this for N>25) (For large system see below)
class Node:
    def __init__(
        self,
        position,
        # data
    ):
        # i, j, k = position
        # self.data = data
        self.position = position  # (i, j, k)
        self.h_coo_p = None
        self.h_coo_m = None

        # positions of neighbors
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.front = None
        self.back = None

    def __repr__(self):
        return f'"Node at {self.position}"'

class Lattice:
    def __init__(self, N, t_x, t_y, m, t_2, w):
        self.N = N
        self.t_x = t_x
        self.t_y = t_y
        self.m = m
        self.t_2 = t_2
        self.w = w
        """This function initializes the lattice and the hamiltonian matrix.
        Args:
        N (int) : size of the lattice
        t_x (float): Hopping in x direction
        t_y (float): Hopping in y direction
        m (float): Onsite energy
        t_2 (float): Hopping in z direction
        w (float): disorder strength
        """
        self.nodes = [
            [
                [
                    None for _ in range(N)
                ] for __ in range(N)
            ] for ___ in range(N)
        ]
        # print(self.nodes)
        self.hamiltonian = np.zeros((2*N**3, 2*N**3))
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    self._make_node(x, y, z)

        for x in range(N):
            for y in range(N):
                for z in range(N):
                    # make the connections and make the hamiltonian
                    # there is a row and a column in the hamiltonian matrix for each baloon of each node
                    # (i, j)th  element is 1 iff the ith baloon can go to the jth baloon
                    # hence it will be a symmetric, sparse matrix
                    node = self.nodes[x][y][z]
                    connection = self.nodes[(x+1)%N][y][z]  # periodic boundary conditions along x
                    node.right = connection
                    connection.left = node
                    self._make_hamiltonian(node, connection, (self.t_x/2, self.t_x/2))

                    connection = self.nodes[x][(y+1)%N][z] # periodic boundary conditions along y
                    node.up = connection
                    connection.down = node
                    self._make_hamiltonian(node, connection, (self.t_y/2, self.t_y/2))
                   
                    connection = self.nodes[x][y][(z+1)%N] # periodic boundary conditions along z
                    node.front = connection
                    node.back = connection
                    self._make_hamiltonian(node, connection, ((1-t_2)/2,(1+t_2)/2))

    def _make_node(self, *position):
        # print(position, self.nodes[position[0]][position[1]][position[2]])
        new_node = Node(position)
        self.nodes[position[0]][position[1]][position[2]] = new_node
        return new_node
    
    def _make_hamiltonian(self, node1, node2, scale=1):
        h_coo_p1, h_coo_m1 = self.h_coo(node1)
        h_coo_p2, h_coo_m2 = self.h_coo(node2)
        # Generate a random number in the range [-w/2, w/2]
        W = np.random.uniform(-self.w/2, self.w/2)
        M = np.random.uniform(-self.w/2, self.w/2)
        # neighbour orbital hopping
        self.hamiltonian[h_coo_p1, h_coo_m2] = 1*scale[0]  # my plus to neighbour minus
        self.hamiltonian[h_coo_m2, h_coo_p1] = 1*scale[0]  # neighbour minus to my plus
        self.hamiltonian[h_coo_m1, h_coo_p2] = 1*scale[1]  # my minus to neighbour plus
        self.hamiltonian[h_coo_p2, h_coo_m1] = 1*scale[1]  # neighbour plus to my minus
        # self orbital hopping
        self.hamiltonian[h_coo_p1, h_coo_m1] = -self.m + W 
        self.hamiltonian[h_coo_m1, h_coo_p1] = -self.m + W 
        self.hamiltonian[h_coo_p2, h_coo_m2] = -self.m + W 
        self.hamiltonian[h_coo_m2, h_coo_p2] = -self.m + W 
         ## on-site energy
        self.hamiltonian[h_coo_p1, h_coo_p1] = M
        self.hamiltonian[h_coo_m1, h_coo_m1] = M
        self.hamiltonian[h_coo_p2, h_coo_p2] = M
        self.hamiltonian[h_coo_m2, h_coo_m2] = M
    def h_coo(self, node):
        """Given a node, returns the position of the node in the hamiltonian matrix.

        Args:
            node (Node): Node of the lattice

        Returns:
            h_coo_p, h_coo_m (int, int): Position of the two baloons in the hamiltonian matrix
        """

        if node.h_coo_p is None or node.h_coo_m is None:
            i, j, k = node.position
            # position of the node in the hamiltonian matrix
            n = (i + j*self.N + k*self.N**2)*2

            # set the position of the node in the hamiltonian matrix
            node.h_coo_p = n
            node.h_coo_m = n+1

        return node.h_coo_p, node.h_coo_m 
    


###====================Discretise the Brillouin zone
class DiscretizeBZ:
    def __init__(self, t_x, t_y, m, t_2):
        """This function is used to initialize the parameters of the model.

        Args:
            t_x (float): Hopping in x direction
            t_y (float): Hopping in y direction
            m (float): Onsite energy
            t_2 (float): Hopping in z direction
            w (float): disorder strength
        """
        self.t_x = t_x
        self.t_y = t_y
        self.m = m
        self.t_2 = t_2
    def calculate_eigen(self, k_points):
        """This function calculates the eigenvalues of the Hamiltonian for different k points in the BZ by discretizing it."""
        y = []  # store the eigenvalues
        k_x_values = []
        k_y_values = []
        k_z_values = []
        q = HamWeyl(self.t_x, self.t_y, self.m, self.t_2)
        for i in range(k_points):
            value = -np.pi + (2 * np.pi * i) / k_points
            k_x_values.append(value)
            k_y_values.append(value)
            k_z_values.append(value)

        k_x_mesh, k_y_mesh, k_z_mesh = np.meshgrid(k_x_values, k_y_values, k_z_values)
        eigenvalues = np.zeros((k_points, k_points, k_points, 2))

        for i in range(k_points):
            for j in range(k_points):
                for k in range(k_points):
                    k_x = k_x_mesh[i, j, k]
                    k_y = k_y_mesh[i, j, k]
                    k_z = k_z_mesh[i, j, k]
                    # Assuming W is an instance of another class that has a method hamiltonian
                    matrix = q.hamiltonian(k_x, k_y, k_z)
                    eigenvalues[i, j, k] = np.linalg.eigvalsh(matrix)
                    eigenvalues[i, j, k] = np.sort(eigenvalues[i, j, k])
                    y.append(eigenvalues[i, j, k][0])
                    y.append(eigenvalues[i, j, k][1])
        return y



###=====================Weyl Nodal Line Hamiltonian in 3d with open boundary conditions
class Node:
    def __init__(
        self,
        position,
        # data
    ):
        # i, j, k = position
        # self.data = data
        self.position = position  # (i, j, k)
        self.h_coo_p = None
        self.h_coo_m = None

        # positions of neighbors
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.front = None
        self.back = None

    def __repr__(self):
        return f'"Node at {self.position}"'

class LatticeOpen:
    def __init__(self, N, t_x, t_y, m, t_2,w):
        self.N = N
        self.t_x = t_x
        self.t_y = t_y
        self.m = m
        self.t_2 = t_2
        self.w = w
        """This function initializes the lattice and the hamiltonian matrix.
        Args:
        N (int) : size of the lattice
        t_x (float): Hopping in x direction
        t_y (float): Hopping in y direction
        m (float): Onsite energy
        t_2 (float): Hopping in z direction
        w (float): disorder strength
        """
        self.nodes = [
            [
                [
                    None for _ in range(N)
                ] for __ in range(N)
            ] for ___ in range(N)
        ]
        # print(self.nodes)
        self.hamiltonian = np.zeros((2*N**3, 2*N**3))
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    self._make_node(x, y, z)

        for x in range(N):
            for y in range(N):
                for z in range(N):
                    # make the connections and make the hamiltonian
                    # there is a row and a column in the hamiltonian matrix for each baloon of each node
                    # (i, j)th  element is 1 iff the ith baloon can go to the jth baloon
                    # hence it will be a symmetric, sparse matrix
                    node = self.nodes[x][y][z]
                    if x < N-1:
                        connection = self.nodes[(x+1)][y][z]  # open boundary conditions along x
                        node.right = connection
                        connection.left = node
                        self._make_hamiltonian(node, connection, (self.t_x/2, self.t_x/2))
                    if y < N-1:
                        connection = self.nodes[x][(y+1)][z] # open boundary conditions along y
                        node.up = connection
                        connection.down = node
                        self._make_hamiltonian(node, connection, (self.t_y/2, self.t_y/2))
                    if z < N-1:
                        connection = self.nodes[x][y][(z+1)] # open boundary conditions along z
                        node.front = connection
                        node.back = connection
                        self._make_hamiltonian(node, connection, ((1-t_2)/2,(1+t_2)/2))

    def _make_node(self, *position):
        # print(position, self.nodes[position[0]][position[1]][position[2]])
        new_node = Node(position)
        self.nodes[position[0]][position[1]][position[2]] = new_node
        return new_node
    
    def _make_hamiltonian(self, node1, node2, scale=1):
        h_coo_p1, h_coo_m1 = self.h_coo(node1)
        h_coo_p2, h_coo_m2 = self.h_coo(node2)
        # Generate a random number in the range [-w, w]
        W = np.random.uniform(-self.w/2, self.w/2)
        M = np.random.uniform(-self.w/2, self.w/2)
        N = np.random.uniform(-self.w/2, self.w/2)
        O = np.random.uniform(-self.w/2, self.w/2)
        # neighbour orbital hopping
        self.hamiltonian[h_coo_p1, h_coo_m2] = 1*scale[0]  # my plus to neighbour minus
        self.hamiltonian[h_coo_m2, h_coo_p1] = 1*scale[0]  # neighbour minus to my plus
        self.hamiltonian[h_coo_m1, h_coo_p2] = 1*scale[1]  # my minus to neighbour plus
        self.hamiltonian[h_coo_p2, h_coo_m1] = 1*scale[1]  # neighbour plus to my minus
        # self orbital hopping
        self.hamiltonian[h_coo_p1, h_coo_m1] = -self.m #+ W/2
        self.hamiltonian[h_coo_m1, h_coo_p1] = -self.m #+ W/2
        self.hamiltonian[h_coo_p2, h_coo_m2] = -self.m #+ W/2
        self.hamiltonian[h_coo_m2, h_coo_p2] = -self.m #+ W/2
        # on-site energy
        self.hamiltonian[h_coo_p1, h_coo_p1] =  W
        self.hamiltonian[h_coo_m1, h_coo_m1] =  M
        self.hamiltonian[h_coo_p2, h_coo_p2] =  N
        self.hamiltonian[h_coo_m2, h_coo_m2] =  O
    def h_coo(self, node):
        """Given a node, returns the position of the node in the hamiltonian matrix.

        Args:
            node (Node): Node of the lattice

        Returns:
            h_coo_p, h_coo_m (int, int): Position of the two baloons in the hamiltonian matrix
        """

        if node.h_coo_p is None or node.h_coo_m is None:
            i, j, k = node.position
            # position of the node in the hamiltonian matrix
            n = (i + j*self.N + k*self.N**2)*2

            # set the position of the node in the hamiltonian matrix
            node.h_coo_p = n
            node.h_coo_m = n+1

        return node.h_coo_p, node.h_coo_m 


###====================================Sparse array to study the large system
class Node:
    def __init__(
        self,
        position,
        # data
    ):
        # i, j, k = position
        # self.data = data
        self.position = position  # (i, j, k)
        self.h_coo_p = None
        self.h_coo_m = None

        # positions of neighbors
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.front = None
        self.back = None

    def __repr__(self):
        return f'"Node at {self.position}"'

class Lattice_S:
    def __init__(self, N, t_x, t_y, m, t_2, w):
        self.N = N
        self.t_x = t_x
        self.t_y = t_y
        self.m = m
        self.t_2 = t_2
        self.w = w
        self.nodes = [
            [
                [
                    None for _ in range(N)
                ] for __ in range(N)
            ] for ___ in range(N)
        ]
        # print(self.nodes)
        # self.hamiltonian = np.zeros((2*N**3, 2*N**3))
        # self.rows = np.zeros(14*N**3)
        # self.cols = np.zeros(14*N**3)
        # self.data = np.zeros(14*N**3)
        # self.count = 0
        self.rows = []
        self.cols = []
        self.data = []
        
        for x in trange(N, desc='Making Nodes'):
            for y in range(N):
                for z in range(N):
                    self._make_node(x, y, z)

        for x in trange(N, desc='Making Hamiltonian'):
            for y in range(N):
                for z in range(N):
                    # make the connections and make the hamiltonian
                    # there is a row and a column in the hamiltonian matrix for each baloon of each node
                    # (i, j)th  element is 1 iff the ith baloon can go to the jth baloon
                    # hence it will be a symmetric, sparse matrix
                    node = self.nodes[x][y][z]
                    connection = self.nodes[(x+1)%N][y][z]  # periodic boundary conditions along x
                    node.right = connection
                    connection.left = node
                    self._make_hamiltonian(node, connection, (self.t_x/2, self.t_x/2))

                    connection = self.nodes[x][(y+1)%N][z] # periodic boundary conditions along y
                    node.up = connection
                    connection.down = node
                    self._make_hamiltonian(node, connection, (self.t_y/2, self.t_y/2))

                    connection = self.nodes[x][y][(z+1)%N] # periodic boundary conditions along z
                    node.front = connection
                    node.back = connection
                    self._make_hamiltonian(node, connection, ((1-t_2)/2, (1+t_2)/2))
        
        self.hamiltonian = coo_array((self.data, (self.rows, self.cols)), shape=(2*N**3, 2*N**3))
        del self.rows, self.cols, self.data

    def _make_node(self, *position):
        # print(position, self.nodes[position[0]][position[1]][position[2]])
        new_node = Node(position)
        self.nodes[position[0]][position[1]][position[2]] = new_node
        return new_node

    def _assign_element(self, i, j, data):
        self.cols.append(i)
        self.rows.append(j)
        self.data.append(data)


    def _make_hamiltonian(self, node1, node2, scale=(1, 1)):
        h_coo_p1, h_coo_m1 = self.h_coo(node1)
        h_coo_p2, h_coo_m2 = self.h_coo(node2)
        # Generate a random number in the range [-w/2, w/2]
        W = np.random.uniform(-self.w/2, self.w/2)
        M = np.random.uniform(-self.w/2, self.w/2)
        # neighbour orbital hopping
        self._assign_element(h_coo_p1, h_coo_m2, 1*scale[0])  # my plus to neighbour minus
        self._assign_element(h_coo_m2, h_coo_p1, 1*scale[0])  # neighbour minus to my plus
        self._assign_element(h_coo_m1, h_coo_p2, 1*scale[1])  # my minus to neighbour plus
        self._assign_element(h_coo_p2, h_coo_m1, 1*scale[1])  # neighbour plus to my minus
        # self orbital hopping
        self._assign_element(h_coo_p1, h_coo_m1, (-self.m + W)/3)  # my plus to my minus
        self._assign_element(h_coo_m1, h_coo_p1, (-self.m + W)/3)  # my minus to my plus
        # self._assign_element(h_coo_p2, h_coo_m2, (-self.m + W)/3)  # neighbour plus to neighbour minus
        # self._assign_element(h_coo_m2, h_coo_p2, (-self.m + W)/3)  # neighbour minus to neighbour plus
        ## on-site energy
        self._assign_element(h_coo_p1, h_coo_p1, M/3)  # my plus to my plus
        self._assign_element(h_coo_m1, h_coo_m1, M/3)  # my minus to my minus
        # self._assign_element(h_coo_p2, h_coo_p2, M/3)  # neighbour plus to neighbour plus
        # self._assign_element(h_coo_m2, h_coo_m2, M/3)  # neighbour minus to neighbour minus


    def h_coo(self, node):
        """Given a node, returns the position of the node in the hamiltonian matrix.

        Args:
            node (Node): Node of the lattice

        Returns:
            h_coo_p, h_coo_m (int, int): Position of the two baloons in the hamiltonian matrix
        """

        if node.h_coo_p is None or node.h_coo_m is None:
            i, j, k = node.position
            # position of the node in the hamiltonian matrix
            n = (i + j*self.N + k*self.N**2)*2

            # set the position of the node in the hamiltonian matrix
            node.h_coo_p = n
            node.h_coo_m = n+1

        return node.h_coo_p, node.h_coo_m 