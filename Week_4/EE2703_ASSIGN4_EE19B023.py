import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
from scipy.linalg import lstsq

#defining the two given functions
def exp(x):
    return np.exp(x)
def coscos(x):
    return np.cos(np.cos(x))
#defining u(x, k) and v(x, k) for both functions
def u1(x, k):
    return exp(x)*np.cos(k*x)
def v1(x, k):
    return exp(x)*np.sin(k*x)
def u2(x, k):
    return coscos(x)*np.cos(k*x)
def v2(x, k):
    return coscos(x)*np.sin(k*x)

#computing fourier coefficients
fcoeff1 = np.zeros(51)
fcoeff2 = np.zeros(51)
fcoeff1[0] = inte.quad(u1, 0, 2*np.pi, args=(0))[0]/(2*np.pi)
fcoeff2[0] = inte.quad(u2, 0, 2*np.pi, args=(0))[0]/(2*np.pi)
for k in range(1, 26):
    fcoeff1[2*k-1] = inte.quad(u1, 0, 2*np.pi, args=(k))[0]/np.pi #a1_k
    fcoeff1[2*k] = inte.quad(v1, 0, 2*np.pi, args=(k))[0]/np.pi #b1_k
    fcoeff2[2*k-1] = inte.quad(u2, 0, 2*np.pi, args=(k))[0]/np.pi #a2_k
    fcoeff2[2*k] = inte.quad(v2, 0, 2*np.pi, args=(k))[0]/np.pi #b2_k

#Least squares method
x = np.linspace(0, 2*np.pi, 401)
x = x[:-1]
mul_factor = np.arange(25) + 1
b1 = exp(x)
b2 = coscos(x)
A = np.zeros((400, 51))
A[:, 0] = 1
A[:, 1::2] = np.cos(mul_factor*(x.reshape(400, 1)))
A[:, 2::2] = np.sin(mul_factor*(x.reshape(400, 1)))
c1 = lstsq(A, b1)[0]
c2 = lstsq(A, b2)[0]

#computing the function value
fun1 = A.dot(c1)
fun2 = A.dot(c2)

#difference in fourier coefficients
err1 = np.max(np.abs(fcoeff1 - c1))
err2 = np.max(np.abs(fcoeff2 - c2))
print(f"The largest error for the function exp(x) is {err1}")
print(f"The largest error for the function cos(cos(x)) is {err2}")

#plot the functions exp(x) and cos(cos(x)) along with the fourier approximations
x1 = np.linspace(-2*np.pi, 4*np.pi, num=300)
xe = np.linspace(0, 2*np.pi, num=100)
plt.figure(num=1)
plt.semilogy(x1, exp(x1), 'r-')
plt.semilogy(x1, np.tile(exp(xe), 3), '--')
plt.title(r'Plot of $e^x$ in the interval $[-2*\pi, 4*\pi)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$e^x$')
plt.legend(['Actual Plot', 'Expected plot from Fourier Expansion'], loc='upper right')
plt.grid(True)
#plt.savefig('fig1.jpeg')
#plt.show()

plt.figure(num=2)
plt.plot(x1, np.cos(np.cos(x1)), 'r-')
plt.plot(x1, np.tile(np.cos(np.cos(xe)), 3), '--')
plt.title(r'Plot of $cos(cos(x))$ in the interval $[-2*\pi, 4*\pi)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$cos(cos(x))$')
plt.legend(['Actual Plot', 'Expected plot from Fourier Expansion'], loc='upper right')
plt.grid(True)
#plt.savefig('fig2.jpeg')
#plt.show()

#plotting the fourier coefficients
n = np.arange(51)
plt.figure(num=3)
plt.semilogy(n, abs(fcoeff1), 'ro')
plt.semilogy(n, abs(c1), 'go', markersize=4.5)
plt.xlabel(r'$n$')
plt.title(r'Semilog plot of fourier coefficients of $e^x$')
plt.legend(['By Integration', 'By Least Squares'], loc='upper right')
plt.grid(True)
#plt.savefig('fig3.jpeg')
#plt.show()
plt.figure(num=4)
plt.loglog(n, abs(fcoeff1), 'ro')
plt.loglog(n, abs(c1), 'go', markersize=4.5)
plt.xlabel(r'$n$')
plt.title(r'Log plot of fourier coefficients of $e^x$')
plt.legend(['By Integration', 'By Least Squares'], loc='upper right')
plt.grid(True)
#plt.savefig('fig4.jpeg')
#plt.show()
plt.figure(num=5)
plt.semilogy(n, abs(fcoeff2), 'ro')
plt.semilogy(n, abs(c2), 'go', markersize=4.5)
plt.xlabel(r'$n$')
plt.title(r'Semilog plot of fourier coefficients of $cos(cos(x))$')
plt.legend(['By Integration', 'By Least Squares'], loc='upper right')
plt.grid(True)
#plt.savefig('fig5.jpeg')
#plt.show()
plt.figure(num=6)
plt.loglog(n, abs(fcoeff2), 'ro')
plt.loglog(n, abs(c2), 'go', markersize=4.5)
plt.xlabel(r'$n$')
plt.title(r'Log plot of fourier coefficients of $cos(cos(x))$')
plt.legend(['By Integration', 'By Least Squares'], loc='upper right')
plt.grid(True)
#plt.savefig('fig6.jpeg')
#plt.show()

plt.figure(num=7)
plt.semilogy(x, fun1, 'go', markersize=4.5)
plt.semilogy(x, exp(x), 'r-')
plt.title(r'$e^x$')
plt.xlabel(r'$x$')
plt.ylabel(r'$e^x$')
plt.legend(['Using Least Squares', 'Actual Value'], loc='upper right')
plt.grid(True)
#plt.savefig('fig7.jpeg')
#plt.show()

plt.figure(num=8)
plt.semilogy(x, fun2, 'go', markersize=4.5)
plt.semilogy(x, coscos(x), 'r-')
plt.title(r'$cos(cos(x))$')
plt.xlabel(r'$x$')
plt.ylabel(r'$cos(cos(x))$')
plt.legend(['Using Least Squares', 'Actual Value'], loc='upper right')
plt.grid(True)
#plt.savefig('fig8.jpeg')
plt.show()