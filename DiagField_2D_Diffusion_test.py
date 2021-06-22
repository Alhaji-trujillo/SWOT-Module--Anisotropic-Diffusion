"""Auxiliary functions used in several notebooks of the repo."""

import numpy as np

Nx, Ny = 100, 80    # Number of grid points in the domain
Lx, Ly = 1000, 800  # Size of the domain in km

def make_grid():
    """Make a 1000 x 800 km^2 grid with Nx x Ny grid points. Returns:
    * Nx, Ny
    * dx, dy: grid steps
    * xm, ym: meshgrid arrays
    """
    dx, dy = Lx/Nx, Ly/Ny   # grid steps, regular, in km
    x, y =np.arange(0,Lx,dx), np.arange(0,Ly,dy) # Zonal and meridional coordinates in km
    ym, xm = np.meshgrid(y,x)
    return dx, dy, xm, ym

dx, dy, xm, ym = make_grid()

def grid_param():
    return dx, dy, Nx, Ny, xm, ym

def make_sla(x,y):
    """Create a 2D field with arbitrary analytical functions"""
#    x,y = x/100, y/100
    f = np.copy(x+y)
    f /= np.max(f)
    return f

noi = np.random.randn(Nx,Ny)

def set_boundaries_to_zero(field):
    f = np.copy(field)
    f[0,:]  = f[-1,:] = f[:,0]  = f[:,-1] = 0
    return f

def f_von_neuman_euler(field, axis=None):
    """Apply Von Neuman boundary conditions to the field."""
    f = np.copy(field)
    if axis == 0 or axis == None:
        f[-1,:] = f[-2,:]
    if axis == 1 or axis == None:
        f[:,-1] = f[:,-2]
    return f

def bk_von_neuman_euler(field, axis=None):
    """Apply Von Neuman boundary conditions to the field."""
    f = np.copy(field)
    if axis == 0 or axis == None:
        f[0,:]  = f[1,:]
    if axis == 1 or axis == None:
        f[:,0]  = f[:,1]
    return f

#def f_derivativeX(field, axis=0): # forward scheme
#    """Compute partial derivative along given axis using forward scheme."""
#    termX = (np.roll(field, -1, axis) - field)/ dx
#    g = f_von_neuman_euler(termX, axis)
#    return g

#def f_derivativeY(field, axis=1): # forward scheme
#    """Compute partial derivative along given axis using forward scheme."""
#    termY = (np.roll(field, -1, axis) - field)/ dy
#    f = f_von_neuman_euler(termY, axis)
#    return f

#def bk_derivativeX(field, axis=0): # backward scheme
#    """Compute partial derivative along given axis using backward scheme."""
#    termX = (field - np.roll(field, 1, axis))/dx
#    g = bk_von_neuman_euler(termX, axis)
#    return g

#def bk_derivativeY(field, axis=1): # backward scheme
#    """Compute partial derivative along given axis using backward scheme."""
#    termY = (field - np.roll(field, 1, axis))/dy
#    f = bk_von_neuman_euler(termY, axis)
#    return f

def gradientX(field):
    """Compute gradient of input scalar field."""
    fx = np.roll(field, -1, axis=0) - field   # Forward scheme
#    fx[-1,:] = field[-4,:] - 3 * field[-3,:] + 3 * field[-2,:]
    return fx

def gradientY(field):
    """Compute gradient of input scalar field."""
    fy = np.roll(field, -1, axis=1) - field
#    fy[:,-1] = field[:,-4] - 3 * field[:,-3] + 3 * field[:,-2] 
    return fy

def divergence(u, v):
    """Compute divergence of a 2D-vector with components u, v."""
    f1 = u - np.roll(u, 1, axis=0)    # Backward Scheme
#    f1[0,:] = u[3,:] - 3 * u[2,:] + 3 * u[1,:]
    f2 = v - np.roll(v, 1, axis=1)
#    f2[:,0] = v[:,3] - 3 * v[:,2] + 3 * v[:,1]
    return f1+f2

def bc2(field):    # Boundary condition option 1
    f = np.copy(field)
    f[0,:]  = 2 * f[1,:] - f[2,:]
    f[:,0]  = 2 * f[:,1] - f[:,2]
    f[-1,:] = 2 * f[-2,:] - f[-3,:]
    f[:,-1] = 2 * f[:,-2] - f[:,-3]
    return f

def bc1(field):    # Boundary Condition Option 2
    f = np.copy(field)
    f[0,:]  = f[1,:]
    f[:,0]  = f[:,1]
    f[-1,:] = f[-2,:]
    f[:,-1] = f[:,-2]
    return f