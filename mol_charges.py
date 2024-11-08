import numpy as np
import ase.build
from scipy.integrate import quad, dblquad,tplquad
import matplotlib.pyplot as plt

uff_radius_qeq = {'H': 0.371,  'O': 0.669}
element_charges =  {'H': 0.4,  'O': -0.8}

def get_phi(alpha):
    return 1/(np.sqrt(2)*alpha)

class molecule():
    def __init__(self,pos,qs,alphas,cell=None):
        self.Nats = len(pos)
        self.positions = pos
        self.qs = qs
        self.alphas = alphas
        self.gauss_charges = []
        if cell is None:
            self.cell = np.zeros(3,3)
            dims = np.max(pos,axis=0)-(np.min(pos,axis=0))
            self.cell[0][0] = dims[0]*1.1
            self.cell[1][1] = dims[1]*1.1
            self.cell[2][2] = dims[2]*1.1
        else:
            self.cell = cell

    def create_gaussians(self):
        for Nat in range(self.Nats):
            self.gauss_charges.append(gaussian3D(self.qs[Nat],self.alphas[Nat],self.positions[Nat]))

    def eval_one(self,Nat,r):
        return self.gauss_charges[Nat].calculate(r)

    def sum_whole(self,x_min,x_max,y_min,y_max,z_min,z_max,bin_size):
        n_x = int((x_max - x_min)/bin_size)
        n_y = int((y_max - y_min)/bin_size)
        n_z = int((z_max - z_min)/bin_size)
        x = np.linspace(x_min, x_max, n_x)
        y = np.linspace(y_min, y_max, n_y)
        z = np.linspace(z_min, z_max, n_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        total_density = np.zeros_like(X)
        for gaussian in self.gauss_charges:
            total_density += gaussian.calculate_arrays(X, Y, Z)

        # gauss_values = self.gauss_charges[0].calculate_arrays(X,Y,Z)
        # dx = (x_max - x_min) / (n_x - 1)
        # dy = (y_max - y_min) / (n_y - 1)
        # dz = (z_max - z_min) / (n_z - 1)
        volume_element = bin_size**3#dx * dy * dz
        integral_approximation = np.sum(total_density) * volume_element
        return integral_approximation

    def scipy_integral(self, x_min, x_max, y_min, y_max, z_min, z_max):
        total_integral = 0.0
        for gaussian in self.gauss_charges:
            integral, error = tplquad(
                gaussian.calculate,
                x_min, x_max,
                lambda x: y_min, lambda x: y_max,
                lambda x, y: z_min, lambda x, y: z_max 
            )
            total_integral += integral
        return total_integral

    def integrate_z(self,z_bin,bin_size):
        n_x = int(self.cell[0][0]/bin_size)
        n_y = int(self.cell[1][1]/bin_size)
        n_z = int(z_bin/bin_size)
        z_lower = 0.0
        n_bins = int(self.cell[2][2]/z_bin)
        bins_integrals = np.zeros(n_bins)
        for n_bin in range(n_bins):
            x = np.linspace(0, self.cell[0][0], n_x)
            y = np.linspace(0, self.cell[1][1], n_y)
            z = np.linspace(z_lower,z_lower+z_bin, n_z)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            total_density = np.zeros_like(X)
            for gaussian in self.gauss_charges:
                total_density += gaussian.calculate_arrays(X, Y, Z)
                # gauss_values = self.gauss_charges[0].calculate_arrays(X,Y,Z)
                volume_element = bin_size**3#dx * dy * dz
                bin_integral = np.sum(total_density) * volume_element
                bins_integrals[n_bin] = bin_integral
            z_lower += z_bin
        return bins_integrals
        '''

        # dx = (x_max - x_min) / (n_x - 1)
        # dy = (y_max - y_min) / (n_y - 1)
        # dz = (z_max - z_min) / (n_z - 1)
        return integral_approximation
        '''
class gaussian3D():
    def __init__(self,q_i,alpha_i, r_i):
        self.alpha = alpha_i
        self.q = q_i
        self.phi = get_phi(self.alpha)
        self.r_i = np.array(r_i)
        self.prevalue = (self.phi/np.sqrt(np.pi))**3 # (self.phi/np.sqrt(np.pi))**3/2

    def calculate(self,r_x,r_y,r_z):
        r = np.array([r_x,r_y,r_z])
        s = r-self.r_i
        N = np.exp(-(self.phi**(2))*np.dot(s,s))
        return self.q*self.prevalue*N

    def calculate_arrays(self,r_x,r_y,r_z):
        s_x = r_x - self.r_i[0]
        s_y = r_y - self.r_i[1]
        s_z = r_z - self.r_i[2]
        s_squared = s_x**2 + s_y**2 + s_z**2
        N = np.exp(-(self.phi**2) * s_squared)
        return -1*self.q * self.prevalue * N

    def integrate(self,x_min,x_max,y_min,y_max,z_min,z_max):
        integral, error = tplquad(
            self.calculate,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            args=()
        )
        return integral


# h2o = ase.build.molecule("H2O")
from ase.io import read
h2o = read("ext_field_res.xyz@4",format="extxyz")
pos_h2o = h2o.get_positions()
alphas_h2o = np.array([uff_radius_qeq[el] for el in h2o.symbols])
# qs_h2o = np.array([element_charges[el] for el in h2o.symbols])
qs_h2o = h2o.arrays["kqeq_charges"]
cell_h2o = np.array(h2o.cell)
my_mol = molecule(pos_h2o,qs_h2o,alphas_h2o,cell_h2o)
my_mol.create_gaussians()

res4 = my_mol.integrate_z(1,0.01)
h2o = read("ext_field_res.xyz@0",format="extxyz")
pos_h2o = h2o.get_positions()
alphas_h2o = np.array([uff_radius_qeq[el] for el in h2o.symbols])
# qs_h2o = np.array([element_charges[el] for el in h2o.symbols])
qs_h2o = h2o.arrays["kqeq_charges"]
cell_h2o = np.array(h2o.cell)
my_mol = molecule(pos_h2o,qs_h2o,alphas_h2o,cell_h2o)
my_mol.create_gaussians()

res0 = my_mol.integrate_z(1,0.01)
plt.plot(range(60),res4-res0)
plt.show()


# alphas_h2o = np.array([uff_radius_qeq["O"],uff_radius_qeq["H"],uff_radius_qeq["H"]])
#qs_h2o = np.array([-0.8,0.4,0.4])

'''
# int_val = my_mol.sum_whole(-0.2,0.2,-0.8,0.8,-0.5,0.2,0.01)
# int_scipy = my_mol.scipy_integral(-0.2,0.2,-0.8,0.8,-0.5,0.2)

int_val = my_mol.sum_whole(-0.2,0.2,-0.2,0.2,-0.2,0.2,0.01)
int_scipy = my_mol.scipy_integral(-0.2,0.2,-0.2,0.2,-0.2,0.2)
# int_val = my_mol.sum_whole(-10,10,-10,10,-10,10,0.1)
# int_scipy = my_mol.scipy_integral(-np.inf,np.inf,-np.inf,np.inf,-np.inf,np.inf)
# for i in range(3):
    # print(my_mol.gauss_charges[i].integrate(-0.2,0.2,-0.8,0.8,-0.5,0.2))
print(int_val,int_val.shape)
print(int_scipy)
'''
