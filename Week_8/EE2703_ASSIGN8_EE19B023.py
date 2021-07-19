import pylab as pl
import scipy.signal as sp
import sympy as sy
import warnings

warnings.filterwarnings("ignore")
sy.init_session

def lowpass(R1, R2, C1, C2, G, Vi):
    s = sy.symbols('s')
    A = sy.Matrix([[0,0,1,-1/G], [-1/(1+s*R2*C2),1,0,0], [0,-G,G,1], [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return A, b, V

def highpass(R1, R2, C1, C2, G, Vi):
    s = sy.symbols('s')
    A = sy.Matrix([[0,-1,0,1/G], [s*C2*R2/(s*C2*R2+1),0,-1,0], [0,G,-G,1], [-1/R1-s*C1-s*C2,0,s*C2,1/R1]])
    b = sy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return A, b, V

def TFconverter(h):
    s = sy.symbols('s')
    n, d = sy.fraction(h)
    N = sy.Poly(n, s).all_coeffs()
    D = sy.Poly(d, s).all_coeffs()
    N, D = [float(f) for f in N], [float(f) for f in D]
    return sp.lti(N, D)

s = sy.symbols('s')

#question 1
A1, b1, V1 = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
h1 = sy.simplify(V1[3])
H1 = TFconverter(h1)
t1 = pl.linspace(0, 1e-3, 10000)
vi1 = pl.ones_like(t1)
t1, Vo1, svec = sp.lsim(H1, vi1, t1)
pl.figure(1)
pl.plot(t1, Vo1)
pl.grid(True)
pl.xlabel(r'$t$')
pl.ylabel(r'$v_o(t)$')
pl.title('Step response of LPF')

#question 2
t2 = pl.linspace(0, 1e-2, 20000)
vi2 = pl.sin(2000*pl.pi*t2) + pl.cos(2e6*pl.pi*t2)
t2, Vo2, svec = sp.lsim(H1, vi2, t2)
pl.figure(2)
pl.plot(t2, Vo2)
pl.grid(True)
pl.xlabel(r'$t$')
pl.ylabel(r'$v_o(t)$')
pl.title('LPF output for sum of sinusoids')

#question 3
A2, b2, V2 = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
h2 = sy.simplify(V2[3])
w = pl.logspace(0, 8, 801)
ss = 1j*w
f = sy.lambdify(s, h2, 'numpy')
pl.figure(3)
pl.loglog(w, pl.absolute(f(ss)), lw=2)
pl.grid(True)
pl.xlabel(r'$\omega$')
pl.ylabel(r'$|H(j\omega)|$')
pl.title('Magnitude Response of HPF')

#question 4
t3 = pl.linspace(0, 1e-3, 30000)
vi3 = pl.exp(-1000*t3)*pl.cos(2e6*pl.pi*t3)
vi4 = pl.exp(-1000*t3)*pl.cos(2e3*pl.pi*t3)
H2 = TFconverter(h2)
t3, Vo3, svec = sp.lsim(H2, vi3, t3)
t3, Vo4, svec = sp.lsim(H2, vi4, t3)
pl.figure(4)
pl.plot(t3, Vo3)
pl.grid(True)
pl.xlabel(r'$t$')
pl.ylabel(r'$v_o(t)$')
pl.title('Response of HPF to a high frequency damped sinusoid')
pl.figure(5)
pl.plot(t3, Vo4)
pl.grid(True)
pl.xlabel(r'$t$')
pl.ylabel(r'$v_o(t)$')
pl.title('Response of HPF to a low frequency damped sinusoid')

#question 5
t1, Vo5, svec = sp.lsim(H2, vi1, t1)
pl.figure(6)
pl.plot(t1, Vo5)
pl.grid(True)
pl.xlabel(r'$t$')
pl.ylabel(r'$v_o(t)$')
pl.title('Step Response of HPF')
pl.show()