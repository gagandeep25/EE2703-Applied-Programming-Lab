import pylab as pl
import sys
from tabulate import tabulate

n = 100
M = 5
Msig = 2
nk = 500
u0 = 5
p = 0.25

if len(sys.argv) != 1:
    try:
        n = sys.argv[1]
        M = sys.argv[2]
        Msig = sys.argv[3]
        nk = sys.argv[4]
        u0 = sys.argv[5]
        p = sys.argv[6]
    except IndexError:
        print("Use the following command to update the values\npython3 <file.py> <n> <M> <Msig> <nk> <u0> <p>")

xx = pl.zeros(n*M)
u = pl.zeros(n*M)
dx = pl.zeros(n*M)
I = []
X = []
V = []

for k in range(nk):
    m = int(pl.randn()*Msig + M)
    nn = pl.where(xx==0)
    xx[nn[0][:m]] = 1
    ii = pl.where(xx>0)
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())
    dx[ii] = u[ii] + 0.5
    xx[ii] += dx[ii]
    u[ii] += 1
    pp = pl.where(xx>n)
    xx[pp] = 0
    u[pp] = 0
    kk = pl.where(u>=u0)[0]
    ll = pl.where(pl.rand(len(kk))<=p)
    kl = kk[ll]
    u[kl] = 0
    xx[kl] -= pl.rand(len(kl))*dx[kl]
    I.extend(xx[kl].tolist())

pl.figure(0)
pl.hist(X, bins=n, edgecolor='black')
pl.xlabel('x')
pl.ylabel('Number of electrons')
pl.title('Electron Density')
pl.savefig('fig1.jpeg')

pl.figure(1)
counts, bins, _ = pl.hist(I, bins=n, range=[0, n], edgecolor='black')
pl.xlabel('x')
pl.ylabel('I')
pl.title('Emission Intensity')
pl.savefig('fig2.jpeg')
xpos = 0.5*(bins[0:-1] + bins[1:])
data = []
[data.append([x, c])  for x, c in zip(xpos, counts)]
f = open('./data.txt', 'w+')
print('Intensity data: \n', file=f)
print(tabulate(data, headers=['xpos', 'count'], tablefmt='orgtbl'), file=f)

pl.figure(2)
pl.plot(X, V, 'x', markersize=5)
pl.xlabel('x')
pl.ylabel('v')
pl.title('Phase Space of electrons')
pl.savefig('fig3.jpeg')
pl.show()