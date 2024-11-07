import numpy as np
from scipy.integrate import quad, dblquad


def get_phi(alpha):
    return 1/(np.sqrt(2)*alpha)

class gaussian2D():
    def __init__(self,alpha, r_xi,r_yi):
        self.alpha = alpha
        self.phi = get_phi(self.alpha)
        self.r_xi = r_xi
        self.r_yi = r_yi
        self.r_i = np.array([r_xi,r_yi])
        self.prevalue = (self.phi/np.sqrt(np.pi))**2 # (self.phi/np.sqrt(np.pi))**3/2

    def calculate_x(self,r_x):
        N = np.exp(-(self.phi**(2))*(r_x-self.r_xi)**2)
        return self.prevalue*N

    def calculate_y(self,r_y):
        N = np.exp(-(self.phi**(2))*(r_y-self.r_yi)**2)
        return self.prevalue*N

    def calculate(self,r_x,r_y):
        r = np.array([r_x,r_y])
        s = r-self.r_i
        N = np.exp(-(self.phi**(2))*np.dot(s,s))
        return self.prevalue*N

    

    def integrate(self,x_min,x_max):
        integral, error = dblquad(
            self.calculate,
            x_min,
            x_max,
            x_min,
            x_max,
            args=()
        )
        return integral

alpha = 0.669  # O in angstroms
r_xi = 0
r_yi = 1
my_gauss = gaussian2D(alpha,r_xi,r_yi)


r_x = np.linspace(-3, 3, 200)
gauss_x = my_gauss.calculate_x(r_x)
r_y = np.linspace(-3, 3, 200)
gauss_y = my_gauss.calculate_y(r_y)


x_min = -np.inf
x_max = np.inf
print("integral",my_gauss.integrate(x_min,x_max))


import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(r_x,gauss_x,marker="o")
ax[1].plot(r_y,gauss_y,marker="o")

plt.show()
