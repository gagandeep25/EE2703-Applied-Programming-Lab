import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.cm as cm
from sys import argv

try:
    Nx = int(argv[1])
    Ny = int(argv[2])
    radius = float(argv[3])
    Niter = int(argv[4])

    phi = pl.zeros((Ny, Nx))
    x = pl.arange(Nx) - (Nx-1)/2
    y = pl.arange(Ny) - (Ny-1)/2
    X, Y = pl.meshgrid(x, y[::-1])
    ii = pl.where(X**2 + Y**2 <= radius**2)
    phi[ii] = 1.0

    pl.figure(1)
    pl.contour(X, Y, phi)
    pl.plot(X[ii], Y[ii], 'ro')
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.title(r"Contour plot of the initial potential")
    #pl.savefig('fig1.jpeg')
    #pl.show()
    errors = pl.zeros(Niter)
    iters = pl.arange(Niter)+1
    for k in range(Niter):
        oldphi = pl.copy(phi)
        phi[1:-1, 1:-1] = 0.25*(phi[:-2, 1:-1] + phi[2:, 1:-1]
                                 + phi[1:-1, :-2] + phi[1:-1, 2:])
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
        phi[0, :] = phi[1, :]
        phi[ii] = 1.0
        errors[k] = pl.amax(pl.absolute(oldphi - phi))

    M1 = pl.ones((Niter, 2))
    M2 = pl.ones((Niter-500, 2))
    M1[:, 1] = iters
    M2[:, 1] = iters[500:]
    sol1 = pl.lstsq(M1, pl.log(errors), rcond=None)[0]
    sol2 = pl.lstsq(M2, pl.log(errors)[500:], rcond=None)[0]
    fit1 = sol1[0] + sol1[1]*iters
    fit2 = sol2[0] + sol2[1]*iters

    pl.figure(2)
    pl.semilogy(iters, pl.exp(fit2), 'k')
    pl.semilogy(iters, errors, 'b')
    pl.semilogy(iters, pl.exp(fit1), 'r')
    pl.xlabel("Number of Iterations")
    pl.legend(['fit2', 'errors', 'fit1'])
    pl.semilogy(iters[::50], errors[::50], 'ro')
    pl.title("Plot of errors vs iterations")
    #pl.savefig('fig2.jpeg')
    #pl.show()

    fig2 = pl.figure(3)
    ax = p3.Axes3D(fig2)
    pl.title(r"The 3D surface plot of $\phi$")
    ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.get_cmap("jet"))
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    #pl.savefig('fig3.jpeg')
    #pl.show()

    pl.figure(4)
    pl.contour(X, Y, phi)
    pl.plot(X[ii], Y[ii], 'ro')
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.title(r"Contour plot of $\phi$")
    #pl.savefig('fig4.jpeg')
    #pl.show()

    #finding and plotting current
    pl.figure(5)
    Jx = pl.zeros_like(phi)
    Jy = pl.zeros_like(phi)
    Jy[1:-1, 1:-1] = 0.5*(phi[2:, 1:-1] - phi[:-2, 1:-1]) #in the document phi_{i-1, j} corresponds to a row below phi_{i+1, j}, whereas here the one with a higher index corresponds to a row below the one with a lower index
    Jx[1:-1, 1:-1] = 0.5*(phi[1:-1, :-2] - phi[1:-1, 2:])
    pl.quiver(X, Y, Jx, Jy, scale=6.5)
    pl.plot(X[ii], Y[ii], 'ro')
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.title(r"Plot of $J$ on the surface of conductor")
    #pl.savefig('fig5.jpeg')
    pl.show()
except IndexError:
    print("Please execute using the following format\npython3 <file.py> <Nx> <Ny> <radius> <Niter>")
except ValueError:
    print("Please make sure that Nx, Ny & Niter are integer values and radius is an integer/float")