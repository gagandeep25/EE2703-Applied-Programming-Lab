import scipy.signal as sp
import pylab as pl

#question 1
X = sp.lti([1, 0.5], [1, 1, 4.75, 2.25, 5.625])
t, x = sp.impulse(X, None, pl.linspace(0, 50, 501))
pl.figure(1)
pl.plot(t, x)
pl.xlabel(r'$t$')
pl.ylabel(r'$x(t)$')

#question 2
X1 = sp.lti([1, 0.05], [1, 0.1, 4.5025, 0.225, 5.068125])
t, x1 = sp.impulse(X1, None, pl.linspace(0, 50, 501))
pl.figure(2)
pl.plot(t, x1)
pl.xlabel(r'$t$')
pl.ylabel(r'$x(t)$')

#question 3
H = sp.lti([1], [1, 0, 2.25])
for w in pl.arange(1.4, 1.65, 0.05):
    f = pl.cos(w*t)*pl.exp(-0.05*t)
    t, x, svec = sp.lsim(H, f, t)
    pl.figure()
    pl.plot(t, x)
    pl.title(f'Plot for frequency {w}')
    pl.xlabel(r'$t$')
    pl.ylabel(r'$x(t)$')

#question 4
X = sp.lti([1, 0, 2], [1, 0, 3, 0])
Y = sp.lti([2], [1, 0, 3, 0])
t, x = sp.impulse(X, None, pl.linspace(0, 20, 501))
t, y = sp.impulse(Y, None, pl.linspace(0, 20, 501))
pl.figure()
pl.plot(t, x)
pl.plot(t, y)
pl.legend([r'$x(t)$', r'$y(t)$'])
pl.xlabel(r'$t$')

#question 5
L = 1e-6; R = 100; C = 1e-6
H = sp.lti([1], [L*C, R*C, 1])
w, mag2, phi = H.bode()
pl.figure()
pl.subplot(2, 1, 1)
pl.semilogx(w, mag2)
pl.xlabel(r'$\omega$')
pl.legend(['Magnitude Response'])
pl.subplot(2, 1, 2)
pl.semilogx(w, phi)
pl.xlabel(r'$\omega$')
pl.legend(['Phase Response'])

#question 6
t = pl.linspace(0, 10e-3, 10001)
vi = pl.cos(1000*t) - pl.cos(1000000*t)
t, vo, svec = sp.lsim(H, vi, t)
pl.figure()
pl.plot(t, vo)
pl.xlabel(r'$t$')
pl.savefig('fig10.jpeg')
pl.figure()
pl.plot(t[:30], vo[:30])
pl.title(r'Plot for t < 30$\mu$s')
pl.show()