import numpy as np
from scipy.integrate import quad


def get_phi(alpha):
    return 1/(np.sqrt(2)*alpha)

class gaussian1D():
    def __init__(self,alpha, r_i):
        self.alpha = alpha
        self.phi = get_phi(self.alpha)
        self.r_i = r_i
        self.prevalue = self.phi/np.sqrt(np.pi) # (self.phi/np.sqrt(np.pi))**3/2

    def calculate(self,r):
        N = np.exp(-(self.phi**(2))*(r-self.r_i)**2)
        return self.prevalue*N

    def integrate(self,x_min,x_max):
        integral, error = quad(
            self.calculate,
            x_min,
            x_max,
            args=()
        )
        return integral
        
alpha = 0.669  # O in angstroms
r_i = 0

my_gauss = gaussian1D(alpha,r_i)

x = np.linspace(-3, 3, 200)
y = my_gauss.calculate(x)

x_min = -np.inf
x_max = np.inf
print("integral",my_gauss.integrate(x_min,x_max))


import matplotlib.pyplot as plt

plt.plot(x,y,marker="o")
plt.show()
