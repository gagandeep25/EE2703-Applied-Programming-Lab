import numpy as np
import matplotlib.pyplot as plt


# Question 1
# y = sin(5*t)
t = np.linspace(0, 2*np.pi, 129)
t = t[0:-1]
y = np.sin(5*t)
Y = np.fft.fftshift(np.fft.fft(y))/128.0
w = np.linspace(-64, 63, 128)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Spectrum of $\sin(5t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
# plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), 'go', lw=2)
plt.xlim([-10, 10])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig1.jpeg')
plt.show()

# y = (1 + 0.1*cos(t))*cos(10*t)
t = np.linspace(-4*np.pi, 4*np.pi, 513)
t = t[0:-1]
y = (1 + 0.1*np.cos(t))*np.cos(10*t)
Y = np.fft.fftshift(np.fft.fft(y))/512.0
w = np.linspace(-64, 64, 513)[:-1]
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-15, 15])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Spectrum of $(1 + 0.1\cos(t))\cos(10t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
# plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
kk = np.where(np.angle(Y) < 1e-6)
phi = np.angle(Y)
phi[kk] = 0
plt.plot(w[ii], phi[ii], 'go', lw=2)
plt.xlim([-15, 15])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig2.jpeg')
plt.show()


# Question 2
# y = sin(t) ** 3
t = np.linspace(-2*np.pi, 2*np.pi, 257)
t = t[0:-1]
y = np.sin(t) ** 3
Y = np.fft.fftshift(np.fft.fft(y))/256.0
w = np.linspace(-64, 63, 256)
plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Spectrum of $\sin^3(t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
# plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), 'go', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig3.jpeg')
plt.show()

# y = cos(t) ** 3
t = np.linspace(-2*np.pi, 2*np.pi, 257)
t = t[0:-1]
y = np.cos(t) ** 3
Y = np.fft.fftshift(np.fft.fft(y))/256.0
w = np.linspace(-64, 63, 256)
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Spectrum of $\cos^3(t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
# plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
kk = np.where(np.angle(Y) < 1e-6)
phi = np.angle(Y)
phi[kk] = 0
plt.plot(w[ii], phi[ii], 'go', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig4.jpeg')
plt.show()


# Question 3
# y = cos(20*t + 5*cos(t))
t = np.linspace(-8*np.pi, 8*np.pi, 1025)
t = t[0:-1]
y = np.cos(20*t + 5*np.cos(t))
Y = np.fft.fftshift(np.fft.fft(y))/1024.0
w = np.linspace(-64, 64, 1025)[:-1]
plt.figure(5)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-35, 35])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Spectrum of $\cos(20t + 5\cos(t))$")
plt.grid(True)
plt.subplot(2, 1, 2)
#plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), 'go', lw=2)
plt.xlim([-35, 35])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig5.jpeg')
plt.show()


# Question 4
T = 2*np.pi
N = 128
tol = 1e-6
err = 1

while True:
    t = np.linspace(-T/2, T/2, N+1)[:-1]
    y = np.exp(-t**2/2)
    y = np.fft.fftshift(y)
    Y = np.fft.fftshift(np.fft.fft(y))*T/N
    w = np.linspace(-N*np.pi/T, N*np.pi/T, N+1)[:-1]
    Yw = np.sqrt(2*np.pi)*np.exp(-w**2/2)
    err = np.absolute(Y - Yw).max()
    if err < tol:
        print(f'The error is {err}')
        print(f'Time window is [{-T/2}, {T/2}]')
        print(f'Number of samples N = {N}')
        break
    T *= 2
    N *= 2

plt.figure(6)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y|$', size=16)
plt.title(r"Estimated Spectrum of Gaussian")
plt.grid(True)
plt.subplot(2, 1, 2)
#plt.plot(w, np.angle(Y), 'ro', lw=2)
ii = np.where(np.absolute(Y) > 1e-3)
kk = np.where(np.angle(Y) < 1e-6)
phi = np.angle(Y)
phi[kk] = 0
plt.plot(w[ii], phi[ii], 'go', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig6.jpeg')
plt.figure(7)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Yw), 'k', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y_{\omega}|$', size=16)
plt.title(r'True Spectrum of Gaussian')
plt.grid(True)
plt.subplot(2, 1, 2)
wii = np.where(np.absolute(Yw) > 1e-3)
wkk = np.where(np.angle(Yw) < 1e-6)
phiw = np.angle(Yw)
phiw[wkk] = 0
plt.plot(w[wii], phi[wii], 'ro', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r"Phase of $Y_{\omega}$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig('fig7.jpeg')
plt.show()
