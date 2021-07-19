'''
Pseudo Code

radius = 1/k; lambda = 2*pi/k
r_bar = grid(3, 3, 3, 1000)
phi = linspace(0, 2*pi, 100)
rprime_bar = radius*c_[cos(phi), sin(phi), zeros_like(phi)]
dlprime_bar = c_[-sin(phi), cos(phi), zeros_like(phi)]*lambda/100
R[i, j, k] = mag(r_bar[i, j, k] - rprime_bar)
A_(x,y)[i, j, k] = sum_l(cos(phi[l])*exp(-1j*k*R[i, j, k, l])*dlprime_bar_(x,y)[l])
B(z) = (A_y[1, 0, z] - A_x[0, 1, z] - A_y[-1, 0, z] + A_x[0, -1, z])/2
plot B(z) vs z
fit [1 log(z)]*[log(c) b] = log(B)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

radius = 10; k = 0.1 #defining the parameters
phi = np.linspace(0, 2*np.pi, 101)[:-1]
rx = np.linspace(0, 2, 3); ry = np.linspace(0, 2, 3); rz = np.linspace(1, 1000, 1000) #components of the r vector
rx, ry, rz = np.meshgrid(rx, ry, rz)
r = np.stack([rx, ry, rz], axis=-1)

#array of rprime vectors indexed by l
rprime = np.array([radius*np.cos(phi), radius*np.sin(phi), np.zeros_like(phi)]).T
#array of dlprime vectors indexed by l
dlprime = np.array([-np.sin(phi)*2*np.pi*radius/100, np.cos(phi)*2*np.pi*radius/100, np.zeros_like(phi)]).T

#plotting the loop
plt.figure(1, figsize=(7, 7))
plt.plot(radius*np.cos(phi), radius*np.sin(phi), 'r.', markersize=10)
plt.quiver(rprime[:, 0], rprime[:, 1], -np.cos(phi)*np.sin(phi), np.cos(phi)*np.cos(phi), color='b', headwidth=7.5)
plt.xlabel(r"$x$ (cm)")
plt.ylabel(r"$y$ (cm)")
plt.title(r"Current Elements")
plt.grid(True)
plt.savefig('fig1.jpg')
#function to calculate R for every point in the volume
def calc(l):
    R = r - rprime[l] #rprime is broadcasted to the shape of r
    R = np.sqrt(np.sum(R**2, axis=-1)) #calculating R
    temp = np.cos(phi[l])*np.exp(-1j*k*R)/R
    temp = np.expand_dims(temp, axis=-1) #This adds another dimension to the array so that the dl vector can be broadcasted
    term = temp*dlprime[l] #extension to find the entire term
    return term

A = np.zeros_like(r, dtype=np.complex128) #initializing the vector potential
for l in range(dlprime.shape[0]): #if l were a vector, there will be issues in broadcasting and hence for loop is used
    A += calc(l)
B = 0.5*(A[1, 2, :, 1] - A[2, 1, :, 0] - A[1, 0, :, 1] + A[0, 1, :, 0]) #calculating B using eqn 2

#plotting the magnetic field
z = np.linspace(1, 1000, 1000)
plt.figure(2)
plt.loglog(z, np.absolute(B), 'k')
plt.xlabel(r"$z$ (cm)")
plt.ylabel(r"$B_z$ (T)")
plt.title(r"Magnetic Field")
plt.grid(True)
plt.savefig('fig2.jpg')
#fitting the magnetic field to the form c*(z**b)
M = np.c_[np.ones_like(z[50:]), np.log(z[50:])] #values starting from the index 50 are taken so that fit is done for the linear region
y = np.log(np.absolute(B[50:]))
x, res, rnk, s = lstsq(M, y)
c = np.exp(x[0]); b = x[1]
print(f"The value of c = {c}\nThe value of b = {b}")

#plotting the fit along with the field
plt.figure(3)
plt.loglog(z, c*(z**b), 'b.', markersize=4)
plt.loglog(z, np.absolute(B), 'k', markersize=4)
plt.legend(['Least Squares Fit for Magnetic Field', 'Magnetic Field'])
plt.xlabel(r"$z$ (cm)")
plt.title("Comparing Magnetic field and its lstsq fit")
plt.grid(True)
plt.savefig('fig3.jpg')
plt.show()