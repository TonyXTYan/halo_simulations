---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python slideshow={"slide_type": ""}
!python -V
```

```python slideshow={"slide_type": ""}
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ["MKL_NUM_THREADS"] = '6' # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = '6' # export NUMEXPR_NUM_THREADS=6
```

```python
import matplotlib.pyplot as plt
import numpy as np
from math import *
from uncertainties import *
from scipy.stats import chi2
import scipy as sp
from matplotlib import gridspec
import matplotlib
import pandas
import sys
import statsmodels.api as sm
import warnings ## statsmodels.api is too old ... -_-#
# from tqdm import tqdm

import pickle
import pgzip
import os, psutil

from IPython.display import display, clear_output

from joblib import Parallel, delayed
N_JOBS=2
from tqdm.notebook import tqdm
# from tqdm import tqdm
from datetime import datetime

import pyfftw
nthreads=2

%config InlineBackend.figure_format = 'retina'
# %config InlineBackend.figure_format = 'svg'
# %config InlineBackend.figure_format = 'pdf'
%matplotlib inline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
# warnings.filterwarnings("ignore", category=UserWarning) 
warnings.formatwarning = lambda s, *args: "Warning: " + str(s)+'\n'

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots
# plt.ion()        # interact with plots without pausing program

```

```python
np.show_config()
```

```python
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
```

```python editable=true slideshow={"slide_type": ""}
from numba import njit, jit, prange, objmode, vectorize
import numba
numba.set_num_threads(8)
```

```python
from numba_progress import ProgressBar
```

```python
from matplotlib.ticker import MaxNLocator
```

```python
import gc
gc.enable(  )
# gc.set_debug(gc.DEBUG_SAVEALL)
```

```python

```

```python
print(np.finfo(np.clongdouble))
print(np.finfo(np.cdouble))
print(np.finfo(np.csingle))
```

```python
dtype = np.cdouble
dtyper = np.float64
```

```python
np.complex128
```

```python

```

```python
use_cache = False
datatime_now = datetime.now()
output_prefix_bracket = "("+datatime_now.strftime("%Y%m%d-%H%M%S") + "-" + \
                        str("T" if use_cache else "F") + ")"
output_prefix = "output/entangled_halo_proj/"+output_prefix_bracket+" "
output_ext = ".pgz.pkl"
print(output_prefix)
```

```python
import platform
if not use_cache:
    with open(output_prefix + "session_info.txt", "wt") as file:
        string = ""
        string += ("="*20 + "Session System Information" + "="*20) + "\n"
        string += "Current Time: "+datatime_now.strftime("%Y %b %d %a - %H:%M:%S")+"\n"
        uname = platform.uname()
        string +=  f"Python Version: {platform.python_version()}\n"
        string +=  f"Platform: {platform.platform()}\n"
        string += (f"System: {uname.system}")+ "\n"
        string += (f"Node Name: {uname.node}")+ "\n"
        string += (f"Release: {uname.release}")+ "\n"
        string += (f"Version: {uname.version}")+ "\n"
        string += (f"Machine: {uname.machine}")+ "\n"
        string += (f"Processor: {uname.processor}")+ "\n"
        string +=  f"CPU Counts: {os.cpu_count()} \n"
    #     print(string)
        file.write(string)
```

## Simulation Parameters

```python
nx = 120+1
nz = 120+1
xmax = 20 # Micrometers
# xmax = 20 * 0.001 # Millimeters
# zmax = (nz/nx)*xmax
zmax = 20
dt = 5e-4 # Milliseconds
# dt = 1e-5 # Seconds
# dt = 0.1 # Microseconds
dx = 2*xmax/(nx-1)
dz = 2*zmax/(nz-1)
hb = 63.5078 #("AtomicMassUnit" ("Micrometers")^2)/("Milliseconds")
# hb = 0.0635077993 # ("AtomicMassUnit" ("Micrometers")^2)/("Microseconds")
# hb = 0.0000635077993 #  ("AtomicMassUnit" ("Millimeters")^2)/("Milliseconds")
m3 = 3   # AtomicMassUnit
m4 = 4 

print("zmax =", zmax)
print("(dx,dz) =", (dx, dz) ) 
print("rotate phase =", 1j*hb*dt/(2*m4*dx*dz)) #want this to be small

pxmax= (nx-1)/2 * 2*pi/(2*xmax)*hb # want this to be greater than p
pzmax= (nz-1)/2 * 2*pi/(2*zmax)*hb
print("pxmax =", pxmax)
print("pzmax =", pzmax)
dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb
print("(dpx,dpz) = ", (dpx, dpz))

print(round((nx*nz)**2/1000/1000,3),"million grid points (ideally want around 1M magnitude)")
print(round(2*8*(nx*nz)**2/1000/1000,3),"MB of data needed estimate")
print(round(5*2*8*(nx*nz)**2/1000/1000,3),"MB of RAM needed to run this thing (estimate)")
print(round((nx*nz)**2*0.001*0.001/60, 2),"minutes/grid_op (for 1μs/element_op)")

assert ((nx*nz)**2 < 1000**4), "This is in the terra range! too big!"

```

```python
2*pi*hb/(2*dx)
```

```python
dpx
```

```python
pxmax/(nx-1)*2
```

```python

```

```python
warnings.warn(str(round(5*2*8*(nx*nz)**2/1000/1000/1000,3)) + " GB of RAM needed to run this thing (estimate)")
```

```python
2*8*(nx*nz)**2/1000/1000/1000
```

```python

```

```python
wavelength = 1.083 #Micrometers
# k = (1/sqrt(2)) * 2*pi / wavelength # effective wavelength
# k = 0.03 * 2*pi / wavelength
k = pi / (2*dx)
# k = pi / (2*dz)
p = 2*hb*k
print("k  =",k,"1/µm")
print("p  =",p, "u*µm/ms")
v3 = hb*k/m3
v4 = hb*k/m4
print("v3 =",v3, "µm/ms")
print("v4 =",v4, "µm/ms")
print("sigma_x =", hb/10, " handwavy heisenberg momentum uncertainty ")
print(2*pi / (2*k), "x-period of cosin bragg lattice")

# sanity check
assert (pxmax > p*2.5 or pzmax > p*2.5), "momentum resolution too small"
assert (dpx < hb/2 and dpz < hb/2), "momentum resolution step too big"
# assert (2*pi / (2*k) > 10*dx), "not even ten x-steps in cosin wave!"
```

```python
# dopd = 60.1025 # 1/ms Doppler detuning (?)
dopd = v3**2 * m3 / hb
print(dopd)
warnings.warn("why? TODO: check")
```

```python

```

```python
pi / (8*dx)
```

```python
a4 = 0.007512 # scattering length 
omega = 50 # I don't know, the nature paper used 50 for Rb
V0 = 2*hb*omega
tBraggPi = np.sqrt(2*pi*hb)/V0
# tBraggCenter = 1
tBraggCenter = tBraggPi * 5
tBraggEnd = tBraggPi * 10
print("V0 =", V0)
print("tBraggPi     =", round(tBraggPi,3),"µs")
print("tBraggCenter =", round(tBraggCenter,3),"µs")
print("tBraggEnd    =", round(tBraggEnd,3),"µs")

def V(t):
    return V0 * (2*pi)**-0.5 * tBraggPi**-1 * np.exp(-0.5*(t-tBraggCenter)**2 * tBraggPi**-2)

def VB(t, tauMid, tauPi):
        return V0 * (2*pi)**-0.5 * tauPi**-1 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)

V0F = 50*1000
def VBF(t, tauMid, tauPi, V0FArg=V0F):
        return V0FArg * (2*pi)**-0.5 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)
    
print(1j*(dt/hb), "term infront of Bragg potential")
print(1j*(dt/hb)*V(tBraggCenter), "max(V)")

pxlin = np.linspace(-pxmax,+pxmax,nx,dtype=dtyper)
pzlin = np.linspace(-pzmax,+pzmax,nz,dtype=dtyper)
```

```python

```

```python
warnings.warn("The code below probably will take " + str(round(5*2*8*(nx*nz)**2/1000/1000/1000,3)) + " GB RAM")

assert 5*2*8*(nx*nz)**2/1000/1000/1000 < 10, " Catch! (stops run all cell - continuing may break the computer!) "
```

```python

```

<!-- #raw -->
psi=np.zeros((nx,nz, nx,nz),dtype=dtype)
<!-- #endraw -->

```python
print(round(psi.nbytes/1000/1000 ,3) , "MB per psi 2d*2d array")
```

```python

```

```python
process = psutil.Process(os.getpid())
def current_py_memory(): return process.memory_info().rss;
def current_py_memory_print(): print(str(round(current_py_memory()/1000**2,3)) + "MB of system memory used")
```

```python
xlin = np.linspace(-xmax,+xmax, nx, dtype=dtyper)
zlin = np.linspace(-zmax,+zmax, nz, dtype=dtyper)
# psi=np.zeros((nx,nz, nx,nz),dtype=dtype)
# psi=np.random.rand(nx,nz, nx,nz) + 1j*np.random.rand(nx,nz, nx,nz)
# print(round(psi.nbytes/1000/1000 ,3) , "MB per psi 2d*2d array")
```

```python
current_py_memory_print()
```

```python

```

```python

```

```python
xgrid = np.tensordot(xlin, np.ones(nz, dtype=dtyper), axes=0)
cosXGrid = np.cos(2*k*xgrid)
# x3grid = np.tensordot(xgrid, np.ones((nx,nz),dtype=dtyper), axes=0)
# x4grid = np.tensordot(np.ones((nx,nz),dtype=dtyper), xgrid, axes=0)
# cosX3Grid = np.cos(2*k*x3grid)
# cosX4Grid = np.cos(2*k*x4grid)

momory_bytes = current_py_memory()
if (momory_bytes/1000**3 > 1): 
    warnings.warn(str(round(momory_bytes/1000**3,3))+" GB of system memory used already! careful!")
else:
    print(momory_bytes/1000**2 , "MB of system memory used")
del momory_bytes
```

<!-- #raw -->
x3grid.dtype
<!-- #endraw -->

```python

```

<!-- #raw -->
expP34Grid = np.zeros((nx,nz,nx,nz),dtype=dtype)
for (iz3, pz3) in enumerate(pzlin):
    for (ix4, px4) in enumerate(pxlin):
        for (iz4, pz4) in enumerate(pzlin):
            expP34Grid[:,iz3,ix4,iz4] = np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin**2 + pz3**2) \
                                               -(1j/hb) * (0.5/m4) * (dt) * (  px4**2 + pz4**2))


# expP3Grid = np.zeros((nx,nz),dtype=dtype)
# for indx in range(nx):
#     expP3Grid[indx, :] = np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin[indx]**2 + pzlin**2)) 
# expP4Grid = np.zeros((nx,nz),dtype=dtype)
# for indx in range(nx):
#     expP4Grid[indx, :] = np.exp(-(1j/hb) * (0.5/m4) * (dt) * (pxlin[indx]**2 + pzlin**2)) 
    
# # expPGrid = np.zeros((nx,nz,nx,nz),dtype=dtype)
# # for iz3 in range(nz):
# #     for ix4 in range(nx):
# #         for iz4 in range(nz):
# #             expPGrid[:, :] = np.exp(-(1j/hb) * (0.5/m4) * (dt) * (pxlin[indx]**2 + pzlin**2)) 

# expP3Grid = np.tensordot(expP3Grid, np.ones((nx,nz)), axes=0)
# expP4Grid = np.tensordot(np.ones((nx,nz)), expP4Grid, axes=0)
<!-- #endraw -->

<!-- #raw -->
del expP3Grid, expP4Grid
<!-- #endraw -->

```python

```

<!-- #raw -->
plt.imshow(np.real(expP34Grid[:,:,0,0].T))
<!-- #endraw -->

<!-- #raw -->
plt.imshow(np.real(expP34Grid[0,:,60,:].T))
<!-- #endraw -->

```python
current_py_memory_print()
```

```python

```

<!-- #raw -->
plt.imshow(np.imag(expP3Grid[:,50,:,50].T))
plt.colorbar()
<!-- #endraw -->

### Contact Potential?

```python
(dx,dz)
```

```python
# # a34 = 0.029 #µm
# a34 = 0.5 #µm
# strength34 = 1e5 # I don't know
```

<!-- #raw -->
expContact = np.zeros((nx,nz, nx,nz),dtype=dtype)
for (iz3, z3) in enumerate(zlin):
    for (ix4, x4) in enumerate(xlin):
        for (iz4, z4) in enumerate(zlin):
            expContact[:,iz3,ix4,iz4] = np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        strength34 *
                                        np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
                                               # inside the guassian contact potential
                                               *0.5*dt
                                        )    
                                        
<!-- #endraw -->

<!-- #raw -->
# not this one
expContact = np.zeros((nx,nz,nx,nz),dtype=dtype)
for (ix3, x3) in enumerate(xlin):
    for (iz3, z3) in enumerate(zlin):
        for (ix4, x4) in enumerate(xlin):
            expContact[ix3,iz3,ix4,:] = np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        strength34 *
                                        np.exp(-((x3-x4)**2 +(z3-zlin)**2)/(4*a34**2))
                                               # inside the guassian contact potential
                                               *0.5*dt
                                        )
<!-- #endraw -->

<!-- #raw -->
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(np.angle(expContact[:,:,60,45]).T)
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(np.flipud(np.angle(expContact[:,45,:,45]).T))
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.flipud(np.angle(expContact[50,:,60,:]).T))
plt.colorbar()
<!-- #endraw -->

<!-- #raw -->
expContact.dtype
<!-- #endraw -->

```python

```

```python
((nx-1)/2,(nz-1)/2)
```

```python

```

```python
current_py_memory_print()
```

```python

```

### Checking parameters

<!-- #raw -->
# smooth bragg in time
tbtest = np.arange(tBraggCenter-5*tBraggPi,tBraggCenter+5*tBraggPi,dt)
# plt.plot(tbtest, V(tbtest))

plt.figure(figsize=(10,2))
plt.plot(tbtest, VBF(tbtest,tBraggPi*5,tBraggPi))
print(np.trapz(V(tbtest),tbtest)) # this should be V0
<!-- #endraw -->

```python
tauPi  = 39*0.0001
tauMid = tauPi*5
tauEnd = tauPi*10

# smooth bragg in time
tbtest = np.arange(0, tauEnd,dt)
# plt.plot(tbtest, V(tbtest))
print(tbtest.size)
plt.figure(figsize=(10,2))
plt.plot(tbtest, VBF(tbtest,tauMid,tauPi))
```

```python
# vtest = np.cos(2*k*xlin)
ncrop = int(0.3*nx)
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(cosXGrid.T)
plt.title("bragg potential grid smooth?")

plt.subplot(2,2,2)
plt.imshow(cosXGrid[:ncrop,:ncrop].T)
plt.title("grid zoomed in")

plt.subplot(2,2,3)
plt.plot(cosXGrid[:,0],alpha=0.9,linewidth=0.5)

plt.subplot(2,2,4)
plt.plot(cosXGrid[:ncrop,0],alpha=0.9,linewidth=0.5)

title="bragg_potential_grid"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show()
del ncrop
del cosXGrid
```

<!-- #raw -->
VBF(tauMid,tauPi,tauMid,V0F)
<!-- #endraw -->

<!-- #raw -->
V0F
<!-- #endraw -->

<!-- #raw -->
VBF(tauMid,tauPi,tauMid,0.0*V0F)
<!-- #endraw -->

<!-- #raw -->
VExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(tauMid, tauPi,tauMid,0.1*V0F) * cosXGrid)
<!-- #endraw -->

<!-- #raw -->
plt.imshow(np.real(VExpGrid.T))
<!-- #endraw -->

<!-- #raw -->

<!-- #endraw -->

<!-- #raw -->
cosXTest = (nx)**-1 * np.trapz(np.trapz(cosX3Grid,dx=dz,axis=3),dx=dx,axis=2)
<!-- #endraw -->

<!-- #raw -->
plt.imshow(cosXTest.T)
<!-- #endraw -->

<!-- #raw -->
plt.plot(cosXTest[:,0])
<!-- #endraw -->

## initial wave function

```python

```

<!-- #raw -->
np.trapz(
    np.trapz(
        np.trapz(
            np.trapz(
                np.abs(psi)**2,
            dx=dz,axis=3),
        dx=dx,axis=2),
    dx=dz,axis=1),
dx=dx,axis=0)
<!-- #endraw -->

<!-- #raw -->
np.trapz(np.trapz(np.trapz(np.trapz(np.abs(psi)**2))))*(dx*dx*dz*dz)
<!-- #endraw -->

```python
dx
```

```python

```

```python
sg = 8;
# @njit(parallel=True, fastmath=True)
# @jit('c16[::1](f8[::1],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,)')
@njit(cache=True)
def psi0gaussianNN(x3, z3, x4, z4, sx3=sg, sz3=sg, sx4=sg, sz4=sg, px3=0.0, pz3=0.0, px4=0.0, pz4=0.0):
    return    np.exp(-0.5*x3**2/sx3**2)\
            * np.exp(-0.5*z3**2/sz3**2)\
            * np.exp(-0.5*x4**2/sx4**2)\
            * np.exp(-0.5*z4**2/sz4**2)\
            * np.exp(+(1j/hb)*(px3*x3 + pz3*z3 + px4*x4 + pz4*z4))

# @njit(parallel=True, fastmath=True)
# @jit('f8(c16[::4],f8,f8)', forceobj=True)
# @jit(forceobj=True)
@njit(cache=True)
def check_norm(psi,dx=dx,dz=dz) -> dtyper:
    return np.trapz(np.trapz(np.trapz(np.trapz(np.abs(psi)**2))))*(dx*dx*dz*dz)
#     return  np.trapz(
#                 np.trapz(
#                     np.trapz(
#                         np.trapz(np.abs(psi)**2,
#                             dx=dz,axis=numba.literally(3)),
#                         dx=dx,axis=numba.literally(2)),
#                     dx=dz,axis=numba.literally(1))
#                 ,dx=dx,axis=numba.literally(0))
#     with objmode(psi='intp[:]',dx='intp[:]',dz='intp[:]'):
#     return np.trapz(np.trapz(np.trapz(np.trapz(np.abs(psi)**2,dx=dz,axis=3),dx=dx,axis=2),dx=dz,axis=1),dx=dx,axis=0)
#     out = -1
#     with objmode(out='intp[:]'):
#         out = np.trapz(np.trapz(np.trapz(np.trapz(np.abs(psi)**2,dx=dz,axis=3),dx=dx,axis=2),dx=dz,axis=1),dx=dx,axis=0)
#     return out
        
# @njit(parallel=True, fastmath=True)
# @njit(parallel=True)

# @njit(parallel=True)
# def psi0gaussian_loop_helper(sx3,sz3,sx4,sz4,px3,pz3,px4,pz4,xlin,zlin):
#     psi=np.zeros((nx,nz, nx,nz),dtype=dtype)
# #     for (iz3, z3) in enumerate(zlin):
#     for iz3 in prange(nz):
#         z3 = zlin[iz3]
#         for (ix4, x4) in enumerate(xlin):
#             for (iz4, z4) in enumerate(zlin):
#                 psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin, z3, x4, z4,sx3,sz3,sx4,sz4,px3,pz3,px4,pz4)
#     return psi

@njit(parallel=True, cache=True)
# @njit(cache=True)
def psi0gaussian(sx3=sg, sz3=sg, sx4=sg, sz4=sg, px3=0, pz3=0, px4=0, pz4=0, xlin=xlin,zlin=zlin) -> np.ndarray:
#     psi =  psi0gaussian_loop_helper(sx3,sz3,sx4,sz4,px3,pz3,px4,pz4,xlin,zlin)
    psi=np.zeros((nx,nz, nx,nz),dtype=dtype)
#     for (iz3, z3) in enumerate(zlin):
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin, z3, x4, z4,sx3,sz3,sx4,sz4,px3,pz3,px4,pz4)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtype)

# @njit(parallel=True)
@jit(forceobj=True)
# @njit
def only3(psi):
    return np.trapz(np.trapz(np.abs(psi)**2 ,dx=dz,axis=3),dx=dx,axis=2)
#     return np.trapz(np.trapz(np.abs(psi)**2,axis=3),axis=2)*(dx*dz)

# @njit(parallel=True)
@jit(forceobj=True)
def only4(psi):
    return np.trapz(np.trapz(np.abs(psi)**2 ,dx=dx,axis=0),dx=dz,axis=0)


```

```python
psi0gaussian.signatures
```

<!-- #raw -->
%timeit psi0gaussian()
<!-- #endraw -->

<!-- #raw -->
_ = psi0gaussian()
<!-- #endraw -->

<!-- #raw -->
psi.dtype
<!-- #endraw -->

<!-- #raw -->
# psi = psi.astype(dtype)
<!-- #endraw -->

<!-- #raw -->

<!-- #endraw -->

<!-- #raw -->
def psi0PairXNN(x3,z3,x4,z4,dx=20,s3=sg,s4=sg,pt=p):
    dx3 = 0.5 * dx;
    dx4 = 0.5 * (m3/m4) * dx;
    ph = 0.5 * pt;
    return (psi0gaussianNN(x3-dx3,z3, x4+dx4,z4, s3,s3, s4,s4, +ph,0,-ph,0) + 
            psi0gaussianNN(x3+dx3,z3, x4-dx4,z4, s3,s3, s4,s4, -ph,0,+ph,0)
           )
def psi0PairX(dx=20,s3=sg,s4=sg,pt=p):
    psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
    for (iz3, z3) in enumerate(zlin):
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0PairXNN(xlin, z3, x4, z4, dx,s3,s4,pt)
    normalisation = check_norm(psi)
    return psi/sqrt(normalisation)

<!-- #endraw -->

```python
# @njit(parallel=True)
# def psi0_just_opposite_loop_helper(dx3,dz3,dx4,dz4,s3,s4,px,pz,xlin,zlin):
#     psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
# #     for (iz3, z3) in enumerate(zlin):
#     for iz3 in prange(nz):
#         z3 = zlin[iz3]
#         for (ix4, x4) in enumerate(xlin):
#             for (iz4, z4) in enumerate(zlin):
#                 psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin-dx3,z3-dz3, x4+dx4,z4+dz4, s3,s3, s4,s4,+px,+pz,-px,-pz)
#     return psi


@njit(parallel=True, cache=True)
# @jit(forceobj=True)
# @njit(cache=True)
def psi0_just_opposite(dr=20,s3=sg,s4=sg,pt=0,a=0,xlin=xlin,zlin=zlin):
    dr3 = 0.5 * dr;
    dr4 = 0.5 * (m3/m4) * dr;
    dx3 = dr3 * cos(a)
    dz3 = dr3 * sin(a)
    dx4 = dr4 * cos(a)
    dz4 = dr4 * sin(a)
    ph = 0.5 * pt;
    px = +ph * cos(a)
    pz = +ph * sin(a)
    
#     print((+px,+pz,-px,-pz))
#     psi = psi0_just_opposite_loop_helper(dx3,dz3,dx4,dz4,s3,s4,px,pz,xlin,zlin)
    psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
#     for (iz3, z3) in enumerate(zlin):
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin-dx3,z3-dz3, x4+dx4,z4+dz4, s3,s3, s4,s4,+px,+pz,-px,-pz)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtype)
    
```

<!-- #raw -->
su=5
psi = psi0_just_opposite(dr=0,s3=su,s4=su,pt=-3*hb*k,a=0*pi)
<!-- #endraw -->

<!-- #raw -->
psi.dtype
<!-- #endraw -->

```python

```

```python

```

```python
@njit(cache=True)
def psi0PairNN(x3,z3,x4,z4,dr=20,s3=sg,s4=sg,pt=p,a=0):
    dr3 = 0.5 * dr;
    dr4 = 0.5 * (m3/m4) * dr;
    dx3 = dr3 * cos(a)
    dz3 = dr3 * sin(a)
    dx4 = dr4 * cos(a)
    dz4 = dr4 * sin(a)
    ph = 0.5 * pt;
    px3 = +ph * cos(a)
    pz3 = +ph * sin(a)
    px4 = -ph * cos(a)
    pz4 = -ph * sin(a)
    return (psi0gaussianNN(x3-dx3,z3-dz3, x4+dx4,z4+dz4, s3,s3, s4,s4,+px3,+pz3,+px4,+pz4) + 
            psi0gaussianNN(x3+dx3,z3+dz3, x4-dx4,z4-dz4, s3,s3, s4,s4,-px3,-pz3,-px4,-pz4)
           )

# @njit(parallel=True)
# def psi0Pair_loop_helper(dr,s3,s4,pt,a,xlin=xlin,zlin=zlin):
#     psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
# #     for (iz3, z3) in enumerate(zlin):
#     for iz3 in prange(nz):
#         z3 = zlin[iz3]
#         for (ix4, x4) in enumerate(xlin):
#             for (iz4, z4) in enumerate(zlin):
#                 psi[:,iz3,ix4,iz4] = psi0PairNN(xlin, z3, x4, z4, dr,s3,s4,pt,a)
#     return psi

@njit(cache=True, parallel=True)
# @njit(cache=True)
def psi0Pair(dr=20,s3=sg,s4=sg,pt=p,a=0,nx=nx,nz=nz):
    psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
#     for (iz3, z3) in enumerate(zlin):
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0PairNN(xlin, z3, x4, z4, dr,s3,s4,pt,a)
    normalisation = check_norm(psi)
    #print(normalisation)
    return (psi/sqrt(normalisation)).astype(dtype)
```

<!-- #raw -->
_ = psi0Pair()
<!-- #endraw -->

<!-- #raw -->
psi.shape
<!-- #endraw -->

```python
@njit(nogil=True,parallel=True,cache=True) # psi0Pair already parallelised
# @njit(nogil=True,cache=True)
# @njit(cache=True)
def psi0ring_loop_helper(dr,s3,s4,pt,an, 
#                         ):
                         progress_proxy=None):
    psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
    angles_list = np.linspace(0,pi,an+1)[:-1]
#     print(an, psi.shape)
    for ia in range(an):
        a = angles_list[ia]
        # print("working on a = "+str(round(a/pi,4))+" * pi      ",end='\r')
#         print(psi0Pair(dr=dr,s3=s3,s4=s4,pt=pt,a=a).shape)
        psi += (1/an) * psi0Pair(dr=dr,s3=s3,s4=s4,pt=pt,a=a,nx=nx,nz=nz)
#         psi_un = psi0Pair_loop_helper(dr=dr,s3=s3,s4=s4,pt=pt,a=a)
        if progress_proxy != None:
            progress_proxy.update(1)
    return psi.astype(dtype)

# @jit(cache=True,forceobj=True)
def psi0ring_with_logging(dr=20,s3=3,s4=3,pt=p,an=4):   
#     psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
#     for (iz3, z3) in enumerate(zlin):
#         for (ix4, x4) in enumerate(xlin):
#             print("working on "+str((iz3,ix4))+"      ",end='\r')
#             for (iz4, z4) in enumerate(zlin):
#                 for a in np.linspace(0,2*pi,360):
#                     psi[:,iz3,ix4,iz4] += psi0PairNN(xlin, z3, x4, z4, dr,s3,s4,pt,a)
#     angles_divs = 4
    angles_list = np.linspace(0,pi,an+1)[:-1]
    
#     if logging:
    print("angles_list = "+ str(np.round(angles_list/pi,4)) + " * pi")
    with ProgressBar(total=an) as progress:
        psi = psi0ring_loop_helper(dr,s3,s4,pt,an,progress)
#     psi = psi0ring_loop_helper(dr,s3,s4,pt,an)
    normalisation = check_norm(psi)
#     print(" ")
#     print("normalisation =", normalisation)
    return (psi/sqrt(normalisation)).astype(dtype)
#     else:
# #         for a in angles_list:
# #             psi += (1/an) * psi0Pair(dr=dr,s3=s3,s4=s4,pt=p,a=a)
#         psi = psi0ring_loop_helper(dr,s3,s4,pt,an,None)
#         normalisation = check_norm(psi)
#         return psi/sqrt(normalisation)
    
# @njit(cache=True)
def psi0ring(dr=20,s3=3,s4=3,pt=p,an=4):   
    psi = psi0ring_loop_helper(dr,s3,s4,pt,an)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtype)
```

```python
_ = psi0ring_with_logging(dr=20,s3=3,s4=3,pt=p,an=1)
_ = psi0ring(dr=20,s3=3,s4=3,pt=p,an=1)
```

```python
gc.collect()
```

<!-- #raw -->
gc.get_count()
<!-- #endraw -->

<!-- #raw -->
%reset out
<!-- #endraw -->

```python

```

```python

```

<!-- #region heading_collapsed=true -->
#### Testings
<!-- #endregion -->

```python hidden=true

```

<!-- #raw hidden=true hide_input=false -->
psi = psi0gaussian(sx3=1, sz3=1, sx4=1, sz4=1, px3=0, pz3=0, px4=0, pz4=0)
print("check normalisation", check_norm(psi))

tempTest3 = only3(psi)
tempTest4 = only4(psi)
plt.subplot(1,2,1)
plt.imshow(np.abs(tempTest3.T)**2)
plt.subplot(1,2,2)
plt.imshow(np.abs(tempTest4.T)**2)
plt.show()

current_py_memory_print()
<!-- #endraw -->

<!-- #raw hidden=true hide_input=false -->
psi = psi0gaussian(sx3=2, sz3=2, sx4=1, sz4=1, px3=0, pz3=0, px4=0, pz4=0)
print("check normalisation", check_norm(psi))

tempTest3 = only3(psi)
tempTest4 = only4(psi)
plt.subplot(1,2,1)
plt.imshow(np.abs(tempTest3.T)**2)
plt.subplot(1,2,2)
plt.imshow(np.abs(tempTest4.T)**2)
plt.show()

current_py_memory_print()
<!-- #endraw -->

<!-- #raw hidden=true hide_input=false -->
su = 3
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)
print("check normalisation", check_norm(psi))

tempTest3 = only3(psi)
tempTest4 = only4(psi)
plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.imshow(tempTest3.T)
# plt.subplot(2,3,2)
# plt.imshow(np.real(tempTest3.T))
# plt.subplot(2,3,3)
# plt.imshow(np.imag(tempTest3.T))

plt.subplot(1,2,2)
plt.imshow(tempTest4.T)
# plt.subplot(2,3,5)
# plt.imshow(np.real(tempTest4.T))
# plt.subplot(2,3,6)
# plt.imshow(np.imag(tempTest4.T))
plt.show()

current_py_memory_print()
<!-- #endraw -->

```python hidden=true
su = 4
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=-100, pz3=-50, px4=100, pz4=50)
t = 0

tempTest3 = only3(psi)
tempTest4 = only4(psi)
print("check normalisation psi", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
tempTest3phi = only3phi(phi)
tempTest4phi = only4phi(phi)
print("check normalisation phi", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(np.flipud(tempTest3.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,2)
plt.imshow(np.flipud(tempTest4.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,3)
plt.imshow((tempTest3phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow((tempTest4phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.show()
del tempTest3, tempTest4, tempTest3phi, tempTest4phi
current_py_memory_print()
```

```python hidden=true

```

```python hidden=true

```

```python hidden=true

```

```python hidden=true
tempTest3phi = only3phi(phi)
```

<!-- #raw hidden=true -->
plt.plot(np.trapz(tempTest3phi,dx=dpz,axis=1))
<!-- #endraw -->

<!-- #raw hidden=true -->
plt.plot(np.trapz(tempTest3phi,dx=dpx,axis=0))
<!-- #endraw -->

```python hidden=true

```

```python hidden=true

```

<!-- #raw hidden=true -->
su = 4
psi = psi0Pair(dr=20,s3=su,s4=su,pt=-hb*k,a=-0.5*pi)
t = 0

tempTest3 = only3(psi)
tempTest4 = only4(psi)
print("check normalisation psi", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
tempTest3phi = only3phi(phi)
tempTest4phi = only4phi(phi)
print("check normalisation phi", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(np.flipud(tempTest3.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,2)
plt.imshow(np.flipud(tempTest4.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,3)
plt.imshow(np.flipud(tempTest3phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(np.flipud(tempTest4phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.show()
del tempTest3, tempTest4, tempTest3phi, tempTest4phi
current_py_memory_print()
<!-- #endraw -->

```python hidden=true

```

```python hidden=true

```

<!-- #region heading_collapsed=true -->
#### Testings v2
<!-- #endregion -->

```python hidden=true
print(22*32/60, "min")
```

```python hidden=true
# su = 4
# psi = psi0Pair(dx=20,s3=su,s4=su,pt=0,a=0)
# del psi
psi = psi0ring_with_logging(dr=5,s3=1.7,s4=1.7,pt=2*hb*k,an=32)
t = 0
```

```python hidden=true
print("check normalisation psi", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
print("check normalisation phi", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,2)
plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,3)
plt.imshow(np.flipud(only3phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(np.flipud(only4phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.show()
current_py_memory_print()
```

```python hidden=true

```

```python hidden=true

```

<!-- #region hidden=true -->
Notes on parameters

`psi = psi0ring(dr=40,s3=2,s4=2,pt=p,an=32,logging=True)` took fucking 15 minutes to generate but it looks pretty smooth

![image.png](attachment:image.png)
<!-- #endregion -->

<!-- #region hidden=true -->
`psi = psi0ring_with_logging(dr=40,s3=3,s4=3,pt=2*hb*k,an=32)`
![image.png](attachment:image.png)
<!-- #endregion -->

```python hidden=true

```

```python hidden=true
((nx-1)/2, (nz-1)/2)
```

```python hidden=true
plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
plt.imshow(np.real(psi[60,:,:,45]))
plt.subplot(2,2,2)
plt.imshow(np.real(psi[:,45,60,:].T))
plt.subplot(2,2,3)
plt.imshow(np.real(psi[60,:,60,:]))
plt.subplot(2,2,4)
plt.imshow(np.real(psi[30,:,90,:]))
```

<!-- #region hidden=true -->
no idea what those mean...
<!-- #endregion -->

```python hidden=true

```

```python hidden=true

```

```python hidden=true

```

## Correlation Function

<!-- #raw -->
ind = abs(zlin) < 5.01*dz
<!-- #endraw -->

<!-- #raw -->
gx3x4 = np.trapz(np.abs(psi[:,:,:,ind])**2,zlin[ind],axis=3)
gx3x4 = np.trapz(gx3x4[:,ind,:],zlin[ind],axis=1)
<!-- #endraw -->

```python
def gx3x4_calc(psi,cut=5.0):
    ind = abs(zlin) < (cut+1e-15)*dz
    gx3x4 = np.trapz(np.abs(psi[:,:,:,ind])**2,zlin[ind],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind,:],zlin[ind],axis=1)
    return gx3x4
```

```python
def plot_gx3x4(gx3x4,cut):
    xip = xlin > 0
    xim = xlin < 0
    gpp = np.trapz(np.trapz(gx3x4[:,xip],xlin[xip],axis=1)[xip],xlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],xlin[xim],axis=1)[xip],xlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],xlin[xip],axis=1)[xim],xlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],xlin[xim],axis=1)[xim],xlin[xim],axis=0)
    E = (gpp+gmm-gpm-gmp)/((gpp+gmm+gpm+gmp))
    
    plt.imshow(np.flipud(gx3x4.T),extent=[-xmax,xmax,-xmax,xmax])
    plt.title("$g^{(2)}_{\pm\pm}$ of $z_\mathrm{cut} = "+str(cut)+"dz$ and $E="+str(round(E,4))+"$")
    plt.xlabel("$x_3$")
    plt.ylabel("$x_4$")
    plt.axhline(y=0,color='white',alpha=0.8,linewidth=0.7)
    plt.axvline(x=0,color='white',alpha=0.8,linewidth=0.7)
    plt.text(+xmax*0.6,+xmax*0.8,"$g^{(2)}_{++}="+str(round(gpp,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-xmax*0.6,+xmax*0.8,"$g^{(2)}_{-+}="+str(round(gmp,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(+xmax*0.6,-xmax*0.8,"$g^{(2)}_{+-}="+str(round(gpm,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-xmax*0.6,-xmax*0.8,"$g^{(2)}_{--}="+str(round(gmm,4))+"$", color='white',ha='center',alpha=0.9)
    
```

```python
plt.figure(figsize=(14,6))

cut_list = [1.0, 5.0, 10.0]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gx3x4_calc(psi,cut=cut)
    plt.subplot(1,3,i+1)
    plot_gx3x4(gx3x4,cut)
plt.show()
```

 eq on p.73 GoodNotes document
 ![image.png](attachment:image.png)

```python
del gx3x4
```

```python

```

```python

```

```python

```

## FFT to momentum

```python
@jit(forceobj=True, cache=True)
def only3phi(phi):
    return np.trapz(np.trapz(np.abs(phi)**2,dx=dpz,axis=3),dx=dpx,axis=2)
#     return np.trapz(np.trapz(np.abs(phi)**2,dx=dpx,axis=2),dx=dpz,axis=2)
@jit(forceobj=True, cache=True)
def only4phi(phi):
    return np.trapz(np.trapz(np.abs(phi)**2,dx=dpx,axis=0),dx=dpz,axis=0)

@njit(cache=True)
def check_norm_phi(phi):
#     return np.trapz(np.trapz(np.trapz(np.trapz(np.abs(phi)**2,dx=dpz,axis=3),dx=dpx,axis=2),dx=dpz,axis=1),dx=dpx,axis=0)
    return np.trapz(np.trapz(np.trapz(np.trapz(np.abs(phi)**2))))*(dpx*dpx*dpz*dpz)


```

<!-- #raw -->
# phiUN = np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(psi,threads=nthreads,norm='ortho')))
phiUN = np.fliplr(np.fft.fftshift(np.fft.fftn(psi,norm='ortho')))
# superWeirdNormalisationFactorSq = np.sum(np.abs(phiUN)**2)*dpx*dpz*dpx*dpz
superWeirdNormalisationFactorSq = check_norm_phi(phiUN)
swnf = sqrt(superWeirdNormalisationFactorSq)
phi = phiUN/swnf
<!-- #endraw -->

```python
@jit(forceobj=True, cache=True)
def phiAndSWNF(psi, nthreads=nthreads):
    phiUN = np.flip(np.flip(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(psi,threads=nthreads,norm='ortho')),axis=1),axis=3)
    superWeirdNormalisationFactorSq = check_norm_phi(phiUN)
    swnf = sqrt(superWeirdNormalisationFactorSq)
    phi = phiUN/swnf
    return phi, (swnf+0*1j)

@jit(forceobj=True, cache=True)
def toPhi(psi, swnf, nthreads=nthreads) -> np.ndarray:
    return np.flip(np.flip(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(psi,threads=nthreads,norm='ortho')),axis=1),axis=3)/swnf

@jit(forceobj=True, cache=True)
def toPsi(phi, swnf, nthreads=nthreads) -> np.ndarray:
    return pyfftw.interfaces.numpy_fft.ifftn(np.fft.ifftshift(np.flip(np.flip(phi*swnf,axis=3),axis=1)),threads=nthreads,norm='ortho')



```

```python
# # @njit(parallel=True,cache=True)
# @jit
# def phiAndSWNF(psi):
#     phiUN = np.flip(np.flip(np.fft.fftshift(np.fft.fftn(psi,norm='ortho')),axis=1),axis=3)
#     superWeirdNormalisationFactorSq = check_norm_phi(phiUN)
#     swnf = sqrt(superWeirdNormalisationFactorSq)
#     phi = phiUN/swnf
#     return phi, swnf

# @njit(parallel=True,cache=True)
# def toPhi(psi, swnf):
#     return np.flip(np.flip(np.fft.fftshift(np.fft.fftn(psi,norm='ortho')),axis=1),axis=3)/swnf

# @njit(parallel=True,cache=True)
# def toPsi(phi, swnf):
#     return np.fft.ifftn(np.fft.ifftshift(np.fliplr(phi*swnf)),norm='ortho')
```

<!-- #raw hide_input=false -->
phi, swnf = phiAndSWNF(psi)
<!-- #endraw -->

```python

```

<!-- #raw hide_input=false -->
check_norm_phi(phi)
<!-- #endraw -->

<!-- #raw hide_input=false -->
su = 5
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)
t = 0

tempTest3 = only3(psi)
tempTest4 = only4(psi)
print("check normalisation", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
tempTest3phi = only3phi(phi)
tempTest4phi = only4phi(phi)
print("check normalisation", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(tempTest3.T)

plt.subplot(2,2,2)
plt.imshow(tempTest4.T)

plt.subplot(2,2,3)
plt.imshow(tempTest3phi.T, extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(tempTest4phi.T, extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.show()
del tempTest3, tempTest4, tempTest3phi, tempTest4phi
current_py_memory_print()
<!-- #endraw -->

```python

```

```python
del t, psi, phi, swnf
```

```python

```

```python

```

## Scattering via contact potential ?

```python
3*hb*k/m3
```

```python
20/249/dt
```

```python

```

<!-- #raw -->
su=3.5
psi = psi0_just_opposite(dr=5,s3=su,s4=su,pt=-2*hb*k,a=0*pi)
t = 0

tempTest3 = only3(psi)
tempTest4 = only4(psi)
print("check normalisation psi", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
tempTest3phi = only3phi(phi)
tempTest4phi = only4phi(phi)
print("check normalisation phi", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(np.flipud(tempTest3.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,2)
plt.imshow(np.flipud(tempTest4.T), extent=[-xmax,xmax,-zmax,zmax])

plt.subplot(2,2,3)
plt.imshow((tempTest3phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow((tempTest4phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# plt.colorbar()

plt.show()
del tempTest3, tempTest4, tempTest3phi, tempTest4phi
current_py_memory_print()
<!-- #endraw -->

<!-- #raw -->
# @njit(parallel=True,cache=True)
@jit(forceobj=True, parallel=True, cache=True)
# @vectorize
def scattering_step_helper(psi,expContact,expP34Grid,swnf) -> np.ndarray:
    psi *= expContact
#     psi = np.multiply(psi,expContact)
#     with objmode(phi='complex128[:]'):
    phi = toPhi(psi,swnf)
    phi *= expP34Grid
#     phi = np.multiply(phi,expP34Grid)
#     with objmode(psi='complex128[:]'):
    psi = toPsi(phi,swnf)
    psi *= expContact
#     psi = np.multiply(psi,expContact)
    return (psi, phi)

#     return toPsi(toPhi(psi*expContact,swnf)*expP34Grid,swnf) * expContact


#     return np.multiply(toPsi(np.multiply(toPhi(np.multiply(psi,expContact),swnf),expP34Grid),swnf),expContact)
    
<!-- #endraw -->

<!-- #raw -->
# numba.set_num_threads(8)
<!-- #endraw -->

<!-- #raw -->
_ = scattering_step_helper(psi0_just_opposite(dr=5,s3=su,s4=su,pt=-2*hb*k,a=0*pi),expContact,expP34Grid,swnf)
<!-- #endraw -->

```python

```

```python code_folding=[]
# del psi_test
```

<!-- #raw -->
@jit(nogil=True, parallel=True, forceobj=True)
def scattering_evolve_loop_helper(t_init, psi_init, swnf, steps=20, progress_proxy=None):
    t = t_init
    psi = psi_init
    #phi = psi_init # just so there's a type reference
    for ia in prange(steps):
        psi *= expContact
#         with objmode(phi='complex128[:,:,:,:]'):
        phi = toPhi(psi,swnf)
        phi *= expP34Grid
#         with objmode(psi='complex128[:,:,:,:]'):
        psi = toPsi(phi,swnf)
        psi *= expContact
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)
    return (t,psi,phi)
<!-- #endraw -->

<!-- #raw -->
@njit(parallel=True,cache=True)
def scattering_evolve_loop_helper2_inner_psi_step(psi_init):
    psi = psi_init
    for (iz3, z3) in enumerate(zlin):
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        strength34 *
                                        np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
                                               # inside the guassian contact potential
                                               *0.5*dt
                                            )
    return psi

@njit(parallel=True,cache=True)
def scattering_evolve_loop_helper2_inner_phi_step(phi_init):
    phi = phi_init
    for (iz3, pz3) in enumerate(pzlin):
        for (ix4, px4) in enumerate(pxlin):
            for (iz4, pz4) in enumerate(pzlin):
                phi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin**2 + pz3**2) \
                                               -(1j/hb) * (0.5/m4) * (dt) * (  px4**2 + pz4**2))
    return phi
                y

# @jit(nogil=True, parallel=True, forceobj=True)
@jit(nogil=True, forceobj=True)
# @njit(nogil=True, parallel=True)
def scattering_evolve_loop_helper2(t_init, psi_init, swnf, steps=20, progress_proxy=None):
    t = t_init
    psi = psi_init
    for ia in prange(steps):
        
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi)
#         phi = np.zeros_like(psi, dtype=dtype)
#         with objmode(phi='complex128[:,:,:,:]'):
        phi = toPhi(psi, swnf)
        phi = scattering_evolve_loop_helper2_inner_phi_step(phi)
        #del psi 
        # might cause memory issues
#         psi2 = np.zeros_like(phi, dtype=dtype)
#         with objmode(psi2='complex128[:,:,:,:]'):
        psi = toPsi(phi, swnf)
#         psi = psi2
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi)
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)                                   
    return (t, psi, phi)
<!-- #endraw -->

```python
numba.set_num_threads(7)
# Each thread will multiply the memory usage!
# e.g. single thread 20GB -> 2 threads ~30GB
```

```python
# a34 = 0.029 #µm
a34 = 0.5 #µm
strength34 = 1e5 # I don't know
```

```python
-(1j/hb) * strength34 * np.exp(-((0-0)**2 +(0-0)**2)/(4*a34**2)) *0.5*dt
```

```python
4*pi*hb*a34/m4
```

```python
strength34/ (4*pi*hb*a34/m4)
```

```python
dpx/(hb*k)
```

```python

```

## Scattering? 

```python
# @njit(parallel=True,cache=True)
@njit
def scattering_evolve_loop_helper2_inner_psi_step(psi_init):
    psi = psi_init
    for iz3 in prange(nz):
        z3 = zlin[iz3]
#     for (iz3, z3) in enumerate(zlin):
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        strength34 *
                                        np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
                                               # inside the guassian contact potential
                                               *0.5*dt
                                            )
    return psi

# @njit(parallel=True,cache=True)
@njit
def scattering_evolve_loop_helper2_inner_phi_step(phi_init):
    phi = phi_init
    for iz3 in prange(nz):
        pz3 = pzlin[iz3]
#     for (iz3, pz3) in enumerate(pzlin):
        for (ix4, px4) in enumerate(pxlin):
            for (iz4, pz4) in enumerate(pzlin):
                phi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin**2 + pz3**2) \
                                               -(1j/hb) * (0.5/m4) * (dt) * (  px4**2 + pz4**2))
    return phi

# @jit(nogil=True, parallel=True, forceobj=True)
@jit(nogil=True, forceobj=True)
# @njit(nogil=True, parallel=True)
def scattering_evolve_loop_helper2(t_init, psi_init, swnf, steps=20, progress_proxy=None):
    t = t_init
    psi = psi_init
    for ia in prange(steps):
        
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi)
#         phi = np.zeros_like(psi, dtype=dtype)
#         with objmode(phi='complex128[:,:,:,:]'):
        phi = toPhi(psi, swnf, nthreads=7)
        phi = scattering_evolve_loop_helper2_inner_phi_step(phi)
        #del psi 
        # might cause memory issues
#         psi2 = np.zeros_like(phi, dtype=dtype)
#         with objmode(psi2='complex128[:,:,:,:]'):
        psi = toPsi(phi, swnf, nthreads=7)
#         psi = psi2
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi)
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)                                   
    return (t, psi, phi)
```

```python

```

```python

```

```python
gc.collect()
su = 1.7
se = 15
print("want to stop at t =", (0.5*se+5)/v3)
print("number of steps is ", (0.5*se+5)/v3/dt)

print_every = 4
frames_count = 50
total_steps = print_every * frames_count
print("Total steps =" ,total_steps)
```

```python
total_steps*dt 
```

```python

```

```python
@njit(parallel=True, cache=True)
def psi0_just_opposite_double(dr=20,s3=sg,s4=sg,pt=0,a=0,xlin=xlin,zlin=zlin,fuck3=0):
    dr3 = 0.5 * dr;
    dr4 = 0.5 * (m3/m4) * dr;
    dx3 = dr3 * cos(a)
    dz3 = dr3 * sin(a)
    dx4 = dr4 * cos(a)
    dz4 = dr4 * sin(a)
    ph = 0.5 * pt;
    px = +ph * cos(a)
    pz = +ph * sin(a)
    
    dx3m = dr3 * cos(-a)
    dz3m = dr3 * sin(-a)
    dx4m = dr4 * cos(-a)
    dz4m = dr4 * sin(-a)
    pxm = +ph * cos(-a)
    pzm = +ph * sin(-a)
    
#     print((+px,+pz,-px,-pz))
#     psi = psi0_just_opposite_loop_helper(dx3,dz3,dx4,dz4,s3,s4,px,pz,xlin,zlin)
    psi = np.zeros((nx,nz,nx,nz),dtype=dtype)
#     for (iz3, z3) in enumerate(zlin):
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = \
                psi0gaussianNN(
                    xlin-dx3,z3-dz3, x4+dx4,z4+dz4, s3,s3, s4,s4,+fuck3*px,+fuck3*pz,-px,-pz) + \
                psi0gaussianNN(
                    xlin-dx3m,z3-dz3m, x4+dx4m,z4+dz4m, s3,s3, s4,s4,+fuck3*pxm,+fuck3*pzm,-pxm,-pzm)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtype)
```

```python
# psi = psi0_just_opposite(dr=0,s3=su,s4=su,pt=-2*hb*k,a=0.5*pi)
# psi = psi0_just_opposite(dr=0,s3=su,s4=su,pt=-2*hb*k,a=+0.5*pi) + \
#       psi0_just_opposite(dr=0,s3=su,s4=su,pt=-2*hb*k,a=-0.5*pi) 
# normalisation = check_norm(psi)
# psi = (psi/sqrt(normalisation)).astype(dtype)
psi = psi0_just_opposite_double(dr=0,s3=su*1.2,s4=su,pt=-2.5*hb*k,a=0.5*pi)

t = 0
f = 0
phi, swnf = phiAndSWNF(psi, nthreads=7)
```

```python

```

tested on A2485
```
%timeit _ = phiAndSWNF(psi,nthreads=1)
28.2 s ± 161 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=2)
21.1 s ± 352 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=3)
18.8 s ± 827 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=4)
17.2 s ± 216 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=5)
16.6 s ± 249 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=6)
12.9 s ± 1.51 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=7)
11.9 s ± 164 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = phiAndSWNF(psi,nthreads=8)
11.8 s ± 114 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```

```python

```

```python

```

```python

```

```python
def scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(current_py_memory()/1000**2,3)) + "MB  ")

    fig = plt.figure(figsize=(12,7))
    plt.subplot(2,2,1)
    plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
#     plt.imshow(np.log(np.flipud(only3(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")
    plt.title("$t="+t_str+" \ ms $")

    plt.subplot(2,2,2)
    plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
#     plt.imshow(np.log(np.flipud(only4(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")

    plt.subplot(2,2,3)
    plt.imshow((only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
#     plt.imshow(np.log(only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
    # plt.colorbar()
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.ylabel("$p_z \ (\hbar k)$")

    plt.subplot(2,2,4)
    plt.imshow((only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
#     plt.imshow(np.log(only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
    # plt.colorbar()
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.xlabel("$p_x \ (\hbar k)$")

    if plt_save:
        title= output_prefix_bracket + " scattering_evolve_loop_plot (f="+str(f)+",t="+t_str+")" 
        plt.savefig("output/"+title+".pdf", dpi=600)
        plt.savefig("output/"+title+".png", dpi=600)
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 
```

```python

```

```python
scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False)
```

```python
gc.collect()
```

```python

```

```python

```

```python
_ = scattering_evolve_loop_helper2(t,psi,swnf,steps=print_every,progress_proxy=None)
```

```python

```

```python code_folding=[]
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=True)
        gc.collect()
        (t,psi,phi) = scattering_evolve_loop_helper2(t,psi,swnf,steps=print_every,progress_proxy=progressbar)
scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=True, plt_save=True)
```

```python

```

```python
def scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=1):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(current_py_memory()/1000**2,3)) + "MB  ")

    fig = plt.figure(figsize=(12,7))
    plt.subplot(2,2,1)
#     plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
#     plt.imshow(np.emath.logn(power,np.flipud(only3(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
#     plt.imshow(np.power(np.flipud(only3(psi).T),power), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
    plt.imshow(np.log(logPlus+np.flipud(only3(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")
    plt.title("$t="+t_str+" \ ms $")

    plt.subplot(2,2,2)
#     plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
#     plt.imshow(np.power(np.flipud(only4(psi).T),power), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
#     plt.imshow(np.emath.logn(power,np.flipud(only4(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
    plt.imshow(np.log(logPlus+np.flipud(only4(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")

    plt.subplot(2,2,3)
#     plt.imshow((only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
#     plt.imshow(np.power(only3phi(phi).T,power), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
#     plt.imshow(np.emath.logn(power,only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
    # plt.colorbar()
    plt.imshow(np.log(logPlus+only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.ylabel("$p_z \ (\hbar k)$")

    plt.subplot(2,2,4)
#     plt.imshow((only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
#     plt.imshow(np.power(only4phi(phi).T,power), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
#     plt.imshow(np.emath.logn(power,only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
    plt.imshow(np.log(logPlus+only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
    # plt.colorbar()
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.xlabel("$p_x \ (\hbar k)$")

    if plt_save:
        title= output_prefix_bracket + " scattering_evolve_loop_plot alt (f="+str(f)+",t="+t_str+",logPlus="+str(logPlus)+")" 
        plt.savefig("output/"+title+".pdf", dpi=600)
        plt.savefig("output/"+title+".png", dpi=600)
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 
```

```python
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=10)
```

```python

```

```python

```

```python
ptt=2.5*hb*k*0.5
pthk = 2.5*0.5
print(ptt,pthk)
```

```python

```

```python
dpz
```

```python
def gp3p4_dhalo_calc(phi,cut=5.0,offset3=0,offset4=0):
#     ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
#     ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
    ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
    ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
#     ind3 = np.logical_or(abs(pzlin-offset3) < (cut+1e-15)*dpz, abs(pzlin+offset3) < (cut+1e-15)*dpz)
#     ind4 = np.logical_or(abs(pzlin-offset4) < (cut+1e-15)*dpz, abs(pzlin+offset4) < (cut+1e-15)*dpz)
    gx3x4 = np.trapz(np.abs(phi[:,:,:,ind4])**2,pzlin[ind4],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind3,:],pzlin[ind3],axis=1)
    return gx3x4 
```

```python
def plot_dhalo_gp3p4(gx3x4,cut,offset3=0,offset4=0):
    xip = pxlin > +0*cut*dpz 
    xim = pxlin < -0*cut*dpz 
    gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
    E = (gpp+gmm-gpm-gmp)/((gpp+gmm+gpm+gmp))
    
    plt.imshow(np.flipud(gx3x4.T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k))
    plt.title("$g^{(2)}_{\pm\pm}$ of $p_\mathrm{cut} = "+str(cut)+"dpz$ and $E="+str(round(E,4))+"$")
    plt.xlabel("$p_3$")
    plt.ylabel("$p_4$")
    plt.axhline(y=0,color='white',alpha=0.8,linewidth=0.7)
    plt.axvline(x=0,color='white',alpha=0.8,linewidth=0.7)
    plt.text(+pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{++}="+str(round(gpp,1))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{-+}="+str(round(gmp,1))+"$", color='white',ha='center',alpha=0.9)
    plt.text(+pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{+-}="+str(round(gpm,1))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{--}="+str(round(gmm,1))+"$", color='white',ha='center',alpha=0.9)
    
```

```python
abs(pzlin-(1.25/4)*(3*8/14)*hb*k) < (0.5+1e-15)*dpz
```

```python
(1.25/4)*(3*8/14)
```

```python

```

```python
plt.figure(figsize=(14,6))

cut_list = [1, 10, 30]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gp3p4_dhalo_calc(psi,cut=cut,offset3=(1.25/4)*(3*8/14)*hb*k,offset4=(1.25/4)*(4*8/14)*hb*k)
    plt.subplot(1,3,i+1)
    plot_dhalo_gp3p4(gx3x4,cut)

title = "double halo corr"
plt.savefig("output/"+title+".pdf", dpi=600)
plt.savefig("output/"+title+".png", dpi=600)

plt.show()
```

```python

```

```python

```

```python
30*dpz/(hb*k)
```

```python

```

```python

```

```python

```

```python
with pgzip.open(output_prefix+'(t,psi)'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump((t,psi), file) 
```

```python

```

```python
with pgzip.open('/Users/tonyyan/Library/CloudStorage/OneDrive-AustralianNationalUniversity/SharePoint - Testing only/(20231014-220227-F) (t,psi).pgz.pkl', 'rb', thread=8) as file:
    (t,psi) = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=8)
```

```python

```

```python

```

```python

```

```python
# https://artmenlope.github.io/plotting-complex-variable-functions/
from colorsys import hls_to_rgb

def colorize(fz):

    """
    The original colorize function can be found at:
    https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
    by the user nadapez.
    """
    
    r = np.log2(1. + np.abs(fz))
    
    h = np.angle(fz)/(2*np.pi)
#     l = 1 - 0.45**(np.log(1+r)) 
    l = 0.75*((np.abs(fz))/np.abs(fz).max())**1.2
    s = 1

    c = np.vectorize(hls_to_rgb)(h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (m,n,3)
    c = np.rot90(c.transpose(2,1,0), 1) # Change shape to (m,n,3) and rotate 90 degrees
    
    return c

```

```python
def gp3p4_dhalo_calc_noAb(phi,cut=5.0,offset3=0,offset4=0):
#     ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
#     ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
    ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
    ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
#     ind3 = np.logical_or(abs(pzlin-offset3) < (cut+1e-15)*dpz, abs(pzlin+offset3) < (cut+1e-15)*dpz)
#     ind4 = np.logical_or(abs(pzlin-offset4) < (cut+1e-15)*dpz, abs(pzlin+offset4) < (cut+1e-15)*dpz)
    gx3x4 = np.trapz(phi[:,:,:,ind4],pzlin[ind4],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind3,:],pzlin[ind3],axis=1)
    return gx3x4 
```

```python
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='cyan', label='$Arg=\pm\pi$', markersize=10, lw=0),
                   Line2D([0], [0], marker='o', color='red', label='$Arg=0$', markersize=10, lw=0)]
```

```python
plt.figure(figsize=(14,9))

# cut_list = [1, 10, 30]
cut_list = [1, 5, 15]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gp3p4_dhalo_calc_noAb(psi,cut=cut,offset3=(1.25/4)*(3*8/14)*hb*k,offset4=(1.25/4)*(4*8/14)*hb*k)
    plt.subplot(2,3,i+1)
    gx3x4_img = colorize(gx3x4.T)
#     plot_dhalo_gp3p4(gx3x4_img,cut)
#     fig, ax = plt.subplots()
    
    xip = pxlin > +0*cut*dpz 
    xim = pxlin < -0*cut*dpz 
    gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
#     E = (gpp+gmm-gpm-gmp)/((gpp+gmm+gpm+gmp))
    
    plt.imshow(gx3x4_img, extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k))
#     ax.imshow(np.flipud(gx3x4_img.T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k))
#     plt.title("$g^{(2)}_{\pm\pm}$ of $p_\mathrm{cut} = "+str(cut)+"dpz$ and $E="+str(round(E,4))+"$")
    plt.title("$g^{(2)}_{\pm\pm}$ of $p_\mathrm{cut} = "+str(cut)+"dpz$")
    plt.xlabel("$p_3$")
    plt.ylabel("$p_4$")
    plt.axhline(y=0,color='white',alpha=0.8,linewidth=0.7)
    plt.axvline(x=0,color='white',alpha=0.8,linewidth=0.7)
#     plt.text(+pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{++}="+str(round(gpp,4))+"$", color='white',ha='center',alpha=0.9)
#     plt.text(-pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{-+}="+str(round(gmp,4))+"$", color='white',ha='center',alpha=0.9)
#     plt.text(+pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{+-}="+str(round(gpm,4))+"$", color='white',ha='center',alpha=0.9)
#     plt.text(-pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{--}="+str(round(gmm,4))+"$", color='white',ha='center',alpha=0.9)
plt.legend(handles=legend_elements, loc='upper right')    

for i in range(3):
    cut = cut_list[i]
    gx3x4 = gp3p4_dhalo_calc(psi,cut=cut,offset3=(1.25/4)*(3*8/14)*hb*k,offset4=(1.25/4)*(4*8/14)*hb*k)
    plt.subplot(2,3,i+1+3)
    plot_dhalo_gp3p4(gx3x4,cut)

# plt.colorbar()
    
plt.tight_layout(pad=0)
title = "double halo corr with phase"
plt.savefig("output/"+title+".pdf", dpi=600)
plt.savefig("output/"+title+".png", dpi=600)

plt.show()
```

```python

```

```python

```

```python

```

```python
N = 3
lim = 3
x, y = np.meshgrid(np.linspace(-lim,lim,N),
                   np.linspace(-lim,lim,N))
z = x + 1j*y
f = (z-1)**5+1j
```

```python
z
```

```python
np.abs(z).max()
```

```python
z.max()
```

```python
colorize(f)
```

```python

```

```python
dx
```

```python
dpz
```

```python
dpz/(hb*k)
```

```python

```

## IDK what's below

```python
import glob
import contextlib
import PIL
```

```python
# https://stackoverflow.com/a/57751793/8798606
# filepaths
fp_in  = "/Users/tonyyan/Documents/_ANU/_Mass-Entanglement/bell correlations/simulations/output/(20230211-114917-F) scattering/*.png"
fp_out = "/Users/tonyyan/Documents/_ANU/_Mass-Entanglement/bell correlations/simulations/output/(20230211-114917-F) scattering/animated.gif"

```

```python
# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(PIL.Image.open(f)) for f in sorted(glob.glob(fp_in)))

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=50, loop=0)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```
print_every = 50
frames_count = 20
total_steps = print_every * frames_count
su = 4
psi = psi0_just_opposite(dr=5,s3=su,s4=su,pt=-2*hb*k,a=0.5*pi)
t = 0.5 		 frame = 19 		 memory used: 8210.645MB  
```

![image-2.png](attachment:image-2.png)

![image.png](attachment:image.png)

```python

```

```python
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        scattering_evolve_loop_plot()
        (t,psi,phi) = scattering_evolve_loop_helper(t,psi,swnf,steps=print_every,progress_proxy=progressbar)
scattering_evolve_loop_plot()
```

```python

```

```python
plt.figure(figsize=(14,6))

cut_list = [5.01, 20.01, 40.01]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gx3x4_calc(psi,cut=cut)
    plt.subplot(1,3,i+1)
    plot_gx3x4(gx3x4,cut)
plt.show()
```

```python

```

```python

```

```python

```

```python

```


![image.png](attachment:image.png)

![image-2.png](attachment:image-2.png)

```python

```

![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)





![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)

```python
with pgzip.open(output_prefix+'(t,psi)'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump((t,psi), file) 
```

```python

```

```python

```

```python

```

<!-- #raw -->
# @jit(forceobj=True, parallel=True, cache=True)
def scattering_evolve_helper(t_init, psi_init, steps=1000):
    # steps = 1000;
    t = t_init
    psi = psi_init
    del psi_init
    phi, swnf = phiAndSWNF(psi)
    for i in tqdm(range(steps)):

        if i % 50 == 0:
            print("t =", round(t,5), "\t\t step =", i, "\t\t memory used: " + 
              str(round(current_py_memory()/1000**2,3)) + "MB  ")

            plt.figure(figsize=(12,6))
            plt.subplot(2,2,1)
            plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax])

            plt.subplot(2,2,2)
            plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax])

            plt.subplot(2,2,3)
            plt.imshow(np.flipud(only3phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
            # plt.colorbar()

            plt.subplot(2,2,4)
            plt.imshow(np.flipud(only4phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
            # plt.colorbar()

            plt.show()
    #     ####
    # #     Vx3ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,1*V0F) * 
    # #                         np.cos(2*k*x3grid + (-10*dopd)*(t-tauMid) + phase) )
    # #     Vx4ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,1*V0F) * 
    # #                         np.cos(2*k*x4grid + (-10*dopd)*(t-tauMid) + phase) )

    #     psi *= Vx3ExpGrid
    #     psi *= Vx4ExpGrid
        psi *= expContact
        phi = toPhi(psi,swnf)
    #     phi *= expP3Grid
    #     phi *= expP4Grid
        phi *= expP34Grid
        psi = toPsi(phi,swnf)
        psi *= expContact
    #     psi *= Vx3ExpGrid
    #     psi *= Vx4ExpGrid

    #     psi = toPsi(toPhi(psi*expContact,swnf)*expP34Grid,swnf) * expContact

#         (psi,phi) = scattering_step_helper(psi,expContact,expP34Grid,swnf)

        t += dt
    return (t,psi)
<!-- #endraw -->

<!-- #raw -->
(t,psi) = scattering_evolve_helper(0, psi0_just_opposite(dr=5,s3=su,s4=su,pt=-2*hb*k,a=0*pi), 1000)
<!-- #endraw -->

```python

```

```python
gx3x4 = gx3x4_calc(psi,cut=cut)
plt.subplot(1,3,i+1)
plot_gx3x4(gx3x4,cut)
plt.show()
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

## Bragg Time Evolution Operator

```python

```

```python

```

<!-- #raw -->
steps = 50;
for i in tqdm(range(steps)):
#     cosX3Grid = np.cos(2*k*x3grid + (-10*dopd)*(t-tauMid) + phase)
#     cosX4Grid = np.cos(2*k*x4grid + (-10*dopd)*(t-tauMid) + phase)
    #Vx3ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,5*V0F) * cosX3Grid )
    #Vx4ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,5*V0F) * cosX4Grid )
    Vx3ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,1*V0F) * 
                        np.cos(2*k*x3grid + (-10*dopd)*(t-tauMid) + phase) )
    Vx4ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,1*V0F) * 
                        np.cos(2*k*x4grid + (-10*dopd)*(t-tauMid) + phase) )
    
    psi *= Vx3ExpGrid
    psi *= Vx4ExpGrid
    phi = toPhi(psi,swnf)
    phi *= expP3Grid
    phi *= expP4Grid
    psi = toPsi(phi,swnf)
    psi *= Vx3ExpGrid
    psi *= Vx4ExpGrid
    
    if steps % 1 == 0:
        print("t =", round(t,5), "\t\t step =", i, "\t\t memory used: " + 
          str(round(current_py_memory()/1000**2,3)) + "MB  ")
        
        plt.figure(figsize=(12,6))
        plt.subplot(2,2,1)
        plt.imshow(only3(psi).T)

        plt.subplot(2,2,2)
        plt.imshow(only4(psi).T)

        plt.subplot(2,2,3)
        plt.imshow(only3phi(phi).T, extent=[-pxmax,pxmax,-pzmax,pzmax])
        # plt.colorbar()

        plt.subplot(2,2,4)
        plt.imshow(only4phi(phi).T, extent=[-pxmax,pxmax,-pzmax,pzmax])
        # plt.colorbar()

        plt.show()
    
    t += dt
<!-- #endraw -->

```python

```

```python code_folding=[]
def numericalEvolve_plotHelper(t, psi, phi):
    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax])

    plt.subplot(2,2,2)
    plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax])

    plt.subplot(2,2,3)
    plt.imshow(np.flipud(only3phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])

    plt.subplot(2,2,4)
    plt.imshow(np.flipud(only4phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])

    plt.show()

def numericalEvolve(
        t_init, 
        psi_init, 
        t_final, 
        tPi  = tBraggPi, 
        tMid = tBraggPi*5, 
        V0FArg=V0F,
        phase  = 0,
        doppd=dopd,
        print_every_t=-1, 
        final_plot=True,
        progress_bar=True, 
    ):
    assert (print_every_t >= dt or print_every_t <= 0), "print_every_t cannot be smaller than dt"
    steps = ceil((t_final - t_init) / dt) 
    t = t_init
    psi = psi_init.copy()
    del psi_init
#     psi = psi_init
    (phi, swnf) = phiAndSWNF(psi)

    def loop():
        nonlocal t
        nonlocal psi
        nonlocal phi
        Vx3ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tMid,tPi,V0FArg) * 
                            np.cos(2*k*x3grid + doppd*(t-tMid) + phase) )
        Vx4ExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tMid,tPi,V0FArg) * 
                            np.cos(2*k*x4grid + doppd*(t-tMid) + phase) )
        psi *= Vx3ExpGrid
        psi *= Vx4ExpGrid
        phi = toPhi(psi,swnf)
        phi *= expP3Grid
        phi *= expP4Grid
        psi = toPsi(phi,swnf)
        psi *= Vx3ExpGrid
        psi *= Vx4ExpGrid
        
        t += dt 
        
    if progress_bar:
        for step in tqdm(range(steps)):
            loop()
            print("finished step =", step, "\t memory used: " + 
                  str(round(current_py_memory()/1000**2,3)) + "MB  ", end='\r');
            if print_every_t > 0 and step % round(print_every_t / dt) == 0: 
                numericalEvolve_plotHelper(t, psi, phi)
    else:
        for step in range(steps):
            loop()
    
    if final_plot:
        print("ALL DONE")
        numericalEvolve_plotHelper(t, psi, phi)
    return (t,psi,phi)
```

```python
su = 5
(t_try1, psi_try1, phi_try1) = numericalEvolve(t_init=0, 
                psi_init=psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0), 
                t_final=tauEnd, tPi=tauPi, tMid=tauMid, V0FArg=1*V0F, phase=0, doppd=-10*dopd, 
                print_every_t=0*dt, final_plot=False, progress_bar=True)
```

Notes

`V0FArg=1*V0F, phase=0, doppd=-10*dopd` doesn't quite work with `tauPi = 0.0039`

`V0FArg=2*V0F, phase=0, doppd=-10*dopd` worked ish

`V0FArg=3*V0F, phase=0, doppd=-10*dopd` worked

`V0FArg=4*V0F, phase=0, doppd=-10*dopd` worked

`V0FArg=10*V0F, phase=0, doppd=-10*dopd` might be too fast

The problem with these are they don't have transfers

then I realised I set the doppler frequency different incorrectly ....

```python

```

```python

```

```python

```

```python
numericalEvolve_plotHelper(t_try1, psi_try1, phi_try1)
```

```python
hbar_k_transfers = np.arange(-6,6+1)
pzlinIndexSet = np.zeros((len(hbar_k_transfers), len(pxlin)), dtype=bool)
# cut_p_width = 0.1
lr_include = 1
for (j, hbar_k) in enumerate(hbar_k_transfers):
#     pzlinIndexSet[j] = abs(pxlin/(hb*k) - hbar_k) <= cut_p_width

    index_unshift = round(hbar_k*(hb*k)/dpx)
    index = int(index_unshift + (nx-1)/2)
    for ishift in range(-lr_include, lr_include+1):
        pzlinIndexSet[j,index+ishift] = True
    
#     print(i,hbar_k, index)
```

```python
# plt.figure()
ax = plt.figure(figsize=(4,4)).gca()
plt.imshow(pzlinIndexSet,interpolation='none',aspect=1, extent=[-pxmax/(hb*k),pxmax/(hb*k),-6,+6])
# plt.axvline(x=(nx-1)/2, linewidth=1, alpha=0.1)
plt.axvline(x=0, linewidth=1, alpha=0.9)
plt.xlabel("$p_x/(\hbar k)$")
plt.ylabel("index accepted")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

title="hbar_k_pxlin_integration_range"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()
```

```python
phiDensityGrid_hbark = np.zeros(len(hbar_k_transfers))
```

```python
phiX3 = np.trapz(np.trapz(np.trapz( (np.abs(phi_try1)**2) , pzlin,axis=3),pxlin,axis=2),pzlin,axis=1)
for (j, hbar_k) in enumerate(hbar_k_transfers):
    index = pzlinIndexSet[j]
    phiDensityGrid_hbark[j] = np.trapz(phiX3[index], pxlin[index])
```

```python
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.plot(phiDensityGrid_hbark,'ko')
plt.subplot(1,2,2)
plt.plot(phiX3)
```

```python

```

```python
np.zeros(10)
```

```python

```

```python
def scan_tauPi_init_gaussian(tPi, a3=0, a4=0, p3=0, p4=0, doppd=dopd, V0FArg=V0F, 
                             logging=False, progress_bar=False):
    tauPi  = tPi
    tauMid = tauPi * 5
    tauEnd = tauPi * 10
    px3 = p3*cos(a3)
    pz3 = p3*sin(a3)
    px4 = p4*cos(a4)
    pz4 = p4*sin(a4)
    
    if logging:
        print("Testing parameters")
        print("tauPi =", round(tPi,6), "    \t tauMid =", round(tauMid,6), " \t tauEnd = ", round(tauEnd,6))
    output = numericalEvolve(t_init=0, 
                psi_init=psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=px3, pz3=pz3, px4=px4, pz4=pz4), 
                t_final=tauEnd, tPi=tauPi, tMid=tauMid, V0FArg=V0FArg, phase=0, doppd=doppd, 
                print_every_t=0*dt, final_plot=False, progress_bar=progress_bar)
    
```

```python
_ = scan_tauPi_init_gaussian(tauPi,a3=0,a4=0,p3=0,p4=0,doppd=-10*dopd,V0FArg=V0F,
                             logging=False,progress_bar=True)
```

```python

```

```python
tPiTest = np.append(np.arange(0.015,0,-dt), 0) # note this is decending
    # tPiTest = np.arange(dt,3*dt,dt)  
```

```python
tPiTest[[1,2,3]]
```

```python
tPiOutput = Parallel(n_jobs=N_JOBS)(
    delayed(lambda tPi: (tPi, scan_tauPi_init_gaussian(tPi,a3=0,a4=0,p3=0,p4=0,doppd=-10*dopd,V0FArg=V0F,
                             logging=False,progress_bar=True)
           )(tPi) 
    for tPi in tqdm(tPiTest)
)     
```

```python
phiDensityGrid = np.zeros((len(tPiTest), pxlin.size))
phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))

for i in tqdm(range(len(tPiTest))):
    item = tPiOutput[i]
    (swnf, phi) = phiAndSWNF(item[1][1])
    phiAbsSq = np.abs(phi)**2
    phiX = np.trapz(phiAbsSq, pzlin,axis=1)
    phiDensityGrid[i] = phiX

    for (j, hbar_k) in enumerate(hbar_k_transfers):
        index = pzlinIndexSet[j]
        phiDensityGrid_hbark[i,j] = np.trapz(phiX[index], pxlin[index])
```

```python

```

```python

```

```python

```

```python

```

```python

```
