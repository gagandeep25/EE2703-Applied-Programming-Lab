import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.axes3d as p3

# Question 1
t = np.linspace(-4*np.pi, 4*np.pi, 257)[:-1]
dt = t[1] - t[0]
fmax = 1/dt
n = np.arange(256)
wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/255))
y = np.sin(np.sqrt(2)*t)*wnd
y[0] = 0
y = np.fft.fftshift(y)
Y = np.fft.fftshift(np.fft.fft(y))/256.0
w = np.linspace(-np.pi*fmax, np.pi*fmax, 257)[:-1]
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.ylabel(r'$|Y|$')
plt.xlim([-5, 5])
plt.title(r'Spectrum of $\sin(\sqrt{2}t)$')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), 'ro', lw=2)
# ii = np.where(np.absolute(Y) > 1e-4)
# plt.plot(w, np.angle(Y), 'go', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'Phase of Y')
plt.xlabel(r'$\omega$')
plt.grid(True)

t1 = np.linspace(-np.pi, np.pi, 65)[:-1]
t2 = np.linspace(-3*np.pi, -np.pi, 65)[:-1]
t3 = np.linspace(np.pi, 3*np.pi, 65)[:-1]
n = np.arange(64)
wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/63))
y = np.sin(np.sqrt(2)*t1)
y1 = y*wnd
plt.figure(2)
plt.plot(t1, y, 'bo')
plt.plot(t2, y, 'ro')
plt.plot(t3, y, 'ro')
plt.xlabel(r'$t$')
plt.title(r'Repeating $\sin(\sqrt{2}t)$ across intevals of $2\pi$')
plt.figure(3)
plt.plot(t1, y1, 'bo')
plt.plot(t2, y1, 'ro')
plt.plot(t3, y1, 'ro')
plt.xlabel(r'$t$')
plt.title(r'Repeating $\sin(\sqrt{2}t)\times w(t)$ across intevals of $2\pi$')


# Question 2
t = np.linspace(-4*np.pi, 4*np.pi, 257)[:-1]
dt = t[1] - t[0]
fmax = 1/dt
w0 = 0.86
y = np.cos(w0*t)**3
y = np.fft.fftshift(y)
Y = np.fft.fftshift(np.fft.fft(y))/256.0
w = np.linspace(-np.pi*fmax, np.pi*fmax, 257)[:-1]
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y|$')
plt.title(r'Spectrum of $\cos^3(0.86t)$ without hamming window')
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), 'ro', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'Phase of Y')
plt.xlabel(r'$\omega$')

t = np.linspace(-4*np.pi, 4*np.pi, 257)[:-1]
dt = t[1] - t[0]
fmax = 1/dt
n = np.arange(256)
wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/255))
w0 = 0.86
y = np.cos(w0*t)**3 * wnd
y = np.fft.fftshift(y)
Y = np.fft.fftshift(np.fft.fft(y))/256.0
w = np.linspace(-np.pi*fmax, np.pi*fmax, 257)[:-1]
plt.figure(5)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$|Y|$')
plt.title(r'Spectrum of $\cos^3(0.86t)$ with hamming window')
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), 'ro', lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'Phase of Y')
plt.xlabel(r'$\omega$')


# Question 3 and 4
t = np.linspace(-np.pi, np.pi, 129)[:-1]
fmax = 1/(t[1] - t[0])
y = np.cos(1.5*t + 0.7) #given
n = np.arange(128)
wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/127))
y1 = y * wnd
y_noise = y + 0.1*np.random.randn(128)
y1 = np.fft.fftshift(y1)
y1_noise = np.fft.fftshift(y_noise * wnd)
Y = np.fft.fftshift(np.fft.fft(y1))/128.0 #dft of y with hamming window
Y_noise = np.fft.fftshift(np.fft.fft(y1_noise))/128.0
w = np.linspace(-np.pi*fmax, np.pi*fmax, 129)[:-1]
plt.figure(6)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.ylabel(r'$|Y|$')
plt.title(r'DFT of $\cos(1.5t + 0.7)$')
plt.xlim([-5, 5])
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), 'ro', lw=2)
plt.xlabel(r'$\omega$')
plt.ylabel(r'Phase of Y')
plt.xlim([-5, 5])

ii = np.where(w >= 0)
sol_w = np.sum(w[ii][:5]*np.absolute(Y)[ii][:5])/np.sum(np.absolute(Y)[ii][:5])
sol_w_noise = np.sum(w[ii][:5]*np.absolute(Y_noise)[ii][:5])/np.sum(np.absolute(Y_noise)[ii][:5])
kk = np.argmax(np.absolute(Y[ii]))
sol_delta = np.angle(Y[ii])[kk]
kk1 = np.argmax(np.absolute(Y_noise[ii]))
sol_delta_noise = np.angle(Y[ii])[kk]
print(sol_w)
print(sol_w_noise)
print(sol_delta)
print(sol_delta_noise)


# Question 5
t = np.linspace(-np.pi, np.pi, 1025)[:-1]
fmax = 1/(t[1] - t[0])
n = np.arange(1024)
wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/1023))
y = np.cos(16*(1.5 + t/(2*np.pi))*t) # chirped signal
y_wnd = y * wnd # chirped signal with window
y = np.fft.fftshift(y)
y_wnd = np.fft.fftshift(y_wnd)
Y = np.fft.fftshift(np.fft.fft(y))/1024.0
Y_wnd = np.fft.fftshift(np.fft.fft(y_wnd))/1024.0
w = np.linspace(-np.pi*fmax, np.pi*fmax, 1025)[:-1]
plt.figure(7)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y), lw=2)
plt.xlim([-75, 75])
plt.title(r'DFT of the Chirped Signal $\cos(16(1.5 + t/2\pi)t)$ without hamming window')
plt.ylabel(r'$|Y|$')
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), 'ro', lw=2)
# ii = np.where(np.absolute(Y) > 1e-4)
# plt.plot(w, np.angle(Y), 'go', lw=2)
plt.xlim([-75, 75])
plt.ylabel(r'Phase of Y')
plt.xlabel(r'$\omega$')
plt.figure(8)
plt.subplot(2, 1, 1)
plt.plot(w, np.absolute(Y_wnd), lw=2)
plt.xlim([-75, 75])
plt.title(r'DFT of the Chirped Signal $\cos(16(1.5 + t/2\pi)t)$ with hamming window')
plt.ylabel(r'$|Y|$')
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y_wnd), 'ro', lw=2)
# ii = np.where(np.absolute(Y1) > 1e-4)
# plt.plot(w, np.angle(Y1), 'go', lw=2)
plt.xlim([-75, 75])
plt.ylabel(r'Phase of Y')
plt.xlabel(r'$\omega$')


# Question 6
t = np.linspace(-np.pi, np.pi, 1025)[:-1]
fmax = 1/(t[1] - t[0])
Y1 = np.zeros((64,16), dtype=np.complex)
Y1_wnd = np.zeros((64, 16), dtype=np.complex)
for i in range(16):
    t1 = t[64*i:(i+1)*64]
    w1 = np.linspace(-np.pi*fmax, np.pi*fmax, 65)[:-1]
    y1 = np.cos(16*(1.5 + t1/(2*np.pi))*t1)
    n = np.arange(64)
    wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/63))
    y1_wnd = y1 * wnd
    y1 = np.fft.fftshift(y1)
    Y1[:, i] = np.fft.fftshift(np.fft.fft(y1))/64.0
    y1_wnd = np.fft.fftshift(y1_wnd)
    Y1_wnd[:, i] = np.fft.fftshift(np.fft.fft(y1_wnd))/64.0
t1 = t[::64]
t1, w1 = np.meshgrid(t1, w1)
fig9 = plt.figure(9)
ax = p3.Axes3D(fig9)
ax.plot_surface(w1, t1, np.absolute(Y1), cmap=cm.jet)
plt.title('DFT Magnitude plot of the Chirped Signal without hamming window')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$t$')
fig10 = plt.figure(10)
ax = p3.Axes3D(fig10)
ax.plot_surface(w1, t1, np.absolute(Y1_wnd), cmap=cm.jet)
plt.title('DFT Magnitude plot of the Chirped Signal with hamming window')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$t$')
plt.show()