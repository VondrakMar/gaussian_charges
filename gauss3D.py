import numpy as np
from scipy.integrate import quad, dblquad,tplquad

def get_phi(alpha):
    return 1/(np.sqrt(2)*alpha)

class gaussian3D():
    def __init__(self,alpha, r_xi,r_yi,r_zi):
        self.alpha = alpha
        self.phi = get_phi(self.alpha)
        self.r_xi = r_xi
        self.r_yi = r_yi
        self.r_zi = r_zi
        self.r_i = np.array([r_xi,r_yi,r_zi])
        self.prevalue = (self.phi/np.sqrt(np.pi))**3 # (self.phi/np.sqrt(np.pi))**3/2

    def calculate_x(self,r_x):
        N = np.exp(-(self.phi**(2))*(r_x-self.r_xi)**2)
        return self.prevalue*N

    def calculate_y(self,r_y):
        N = np.exp(-(self.phi**(2))*(r_y-self.r_yi)**2)
        return self.prevalue*N

    def calculate(self,r_x,r_y,r_z):
        r = np.array([r_x,r_y,r_z])
        s = r-self.r_i
        N = np.exp(-(self.phi**(2))*np.dot(s,s))
        return self.prevalue*N

    def integrate(self,x_min,x_max):
        integral, error = tplquad(
            self.calculate,
            x_min,
            x_max,
            x_min,
            x_max,
            x_min,
            x_max,
            args=()
        )
        return integral

alpha = 0.669  # O in angstroms
r_xi = 0
r_yi = 0
r_zi = 0
my_gauss = gaussian3D(alpha,r_xi,r_yi,r_zi)


r_x = np.linspace(-3, 3, 200)
gauss_x = my_gauss.calculate_x(r_x)
r_y = np.linspace(-3, 3, 200)
gauss_y = my_gauss.calculate_y(r_y)


x_min = -np.inf
x_max = np.inf
print("integral",my_gauss.integrate(x_min,x_max))


# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].plot(r_x,gauss_x,marker="o")
# ax[1].plot(r_y,gauss_y,marker="o")

# plt.show()
