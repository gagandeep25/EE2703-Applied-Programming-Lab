#Make sure that fitting.dat file is in the same directory as this file
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq

def g(t, A, B):
    return A*sp.jv(2, t) + B*t

data = np.loadtxt('fitting.dat')
sigma = np.logspace(-1, -3, 9)
t = data[:, 0]
names = []

#plot for question 4
plt.figure(num=0)
for i in range(1, data.shape[1]):
    plt.plot(t, data[:, i])
    names.append(r'$\sigma_%d$ = %.3f' % (i, sigma[i-1]))
plt.plot(t, g(t, 1.05, -0.105), color='k')
names.append('True Value')
plt.legend(names)
plt.xlabel(r'$t \longrightarrow$')
plt.ylabel(r'$f(t) + noise \longrightarrow$')
plt.title(r'Q4: Data to be fitted to theory')
plt.show()

#plot for question 5
plt.figure(num=1)
plt.errorbar(t[::5], data[::5, 1], 0.10, fmt='ro')
plt.plot(t, g(t, 1.05, -0.105))
plt.legend([r'$f(t)$', 'Error Bar'])
plt.xlabel(r'$t \longrightarrow$')
plt.title(r'Q5: Data points for $\sigma = %.2f$ along with exact function' % sigma[0])
plt.show()

M = np.c_[sp.jv(2, t), t]
#print(np.allclose(g(t, 1.05, -0.105), M.dot(np.array([1.05, -0.105])))) #check if two arrays are equal
A = np.arange(0, 2.1, 0.1)
B = np.arange(-0.2, 0.01, 0.01)
msqerror = np.zeros((A.shape[0], B.shape[0]))
for i in range(A.shape[0]):
    for j in range(B.shape[0]):
        msqerror[i, j] = np.sum(np.square(data[:, 1] - g(t, A[i], B[j]))) #computation of mean square error
msqerror /= 101

#plot for question 8
ind = np.unravel_index(np.argmin(msqerror), msqerror.shape)
fig, ax = plt.subplots()
CS = ax.contour(A, B, msqerror, levels=16)
ax.clabel(CS, CS.levels[:5], inline=1, fontsize=10)
ax.plot(A[ind[0]], B[ind[1]], marker='o', color='r')
ax.annotate("Location of minimum", xy=(A[ind[0]], B[ind[1]]))
plt.xlabel(r'$A \longrightarrow$')
plt.ylabel(r'$B \longrightarrow$')
plt.title(r'Q8: Contour plot of $\epsilon_{ij}$')
plt.show()

A_pred = np.zeros(data.shape[1]-1)
B_pred = np.zeros(data.shape[1]-1)
for i in range(1, data.shape[1]):
    A_pred[i-1], B_pred[i-1] = lstsq(M, data[:, i])[0] #least square estimates for all columns of data

#plot for question 10
plt.figure(num=3)
plt.plot(sigma, np.abs(A_pred - 1.05), 'o--')
plt.plot(sigma, np.abs(B_pred - (-0.105)), 'o--')
plt.legend(['Aerr', 'Berr'])
plt.xlabel(r'Noise Standard Deviation $\longrightarrow$')
plt.ylabel(r'Error in estimation of A and B $\longrightarrow$')
plt.title(r'Q10: Variation of error with noise')
plt.show()

#plot for question 11
plt.figure(num=4)
plt.stem(sigma, np.abs(A_pred - 1.05), use_line_collection=True)
plt.stem(sigma, np.abs(B_pred - (-0.105)), use_line_collection=True)
plt.loglog(sigma, np.abs(A_pred - 1.05), 'ro')
plt.loglog(sigma, np.abs(B_pred - (-0.105)), 'bo')
plt.legend(['Aerr', 'Berr'])
plt.xlabel(r'$\sigma_n \longrightarrow$')
plt.ylabel(r'Error in estimation of A and B $\longrightarrow$')
plt.title(r'Q11: Variation of error with noise log plot')
plt.show()