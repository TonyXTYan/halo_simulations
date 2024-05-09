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

# Two Particles


# Setting things up


## Packages

```python
!python -V
```

```python
import matplotlib.pyplot as plt
import numpy as np
from math import *
from uncertainties import *
from scipy.stats import chi2
import scipy
from matplotlib import gridspec
import matplotlib
import pandas as pd
import sys
import statsmodels.api as sm
import warnings ## statsmodels.api is too old ... -_-#

import pickle
import pgzip
import os
import platform
import logging
import sys
import psutil
import glob
import re
from moviepy.editor import ImageSequenceClip
import cv2

from IPython.display import display, clear_output, HTML

from joblib import Parallel, delayed

from tqdm.notebook import tqdm
# from tqdm import tqdm
from datetime import datetime
import time

import pyfftw


%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots
```

```python
from numba import njit, jit, prange, objmode, vectorize
import numba
numba.set_num_threads(8)
from numba_progress import ProgressBar
from matplotlib.ticker import MaxNLocator
```

```python
N_JOBS=2#-3-1
nthreads=2
```

```python
# np.show_config()
```

```python
import gc
gc.enable(  )
```

```python
use_cache = False
save_cache = False
save_debug = True 

datetime_init = datetime.now()

# os.makedirs('output/oneParticleSim', exist_ok=True)
output_prefix = "output/twoParticleSim/"+\
                datetime_init.strftime("%Y%m%d-%H%M%S") + "-" + \
                str("T" if save_debug else "F") + \
                str("T" if use_cache else "F") + \
                str("T" if save_cache else "F") + \
                "/"
output_ext = ".pgz.pkl"
os.makedirs(output_prefix, exist_ok=True)
print(output_prefix)
```

```python
plt.set_loglevel("warning")
```

```python
l = logging.getLogger()
l.setLevel(logging.NOTSET)
l_formatter = logging.Formatter('%(asctime)s - %(levelname)s - \n%(message)s')
l_file_handler = logging.FileHandler(f'{output_prefix}/logs.log', encoding='utf-8')
l_file_handler.setFormatter(l_formatter)
l.addHandler(l_file_handler)
l_console_handler = logging.StreamHandler(sys.stdout)
l_console_handler.setLevel(logging.INFO)
l_console_handler.setFormatter(logging.Formatter('%(message)s'))
l.addHandler(l_console_handler)
```

```python
# if save_debug:
    # with open(output_prefix + "session_info.txt", "wt") as file:
s = ""
s += ("="*20 + "Session System Information" + "="*20) + "\n"
uname = platform.uname()
s +=  f"Python Version: {platform.python_version()}\n"
s +=  f"Platform: {platform.platform()}\n"
s += (f"System: {uname.system}")+ "\n"
s += (f"Node Name: {uname.node}")+ "\n"
s += (f"Release: {uname.release}")+ "\n"
s += (f"Version: {uname.version}")+ "\n"
s += (f"Machine: {uname.machine}")+ "\n"
s += (f"Processor: {uname.processor}")+ "\n"
s +=  f"CPU Counts: {os.cpu_count()} \n"
        # print(string)
        # file.write(string)
    # print(string)
l.info(s)

l.info(f"""nthreads = {nthreads}
N_JOBS = {N_JOBS}""")
```

```python
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_ram_usage(items=globals().items(), cutoff=10):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(items))
                             , key= lambda x: -x[1])[:cutoff]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
```

```python
print_ram_usage()
# print_ram_usage(locals().items(),10)
# print_ram_usage(globals().items(),10)
```

```python

```

## Simulation Parameters

```python
dtypec = np.cdouble
dtyper = np.float64
l.info(f"dtypec = {dtypec}, dtyper = {dtyper}")
```

```python
nx = 120+1
nz = 120+1
xmax = 30 #Micrometers
zmax = (nz/nx)*xmax
# zmax = xmax
dt = 1e-4 # Milliseconds
dx = 2*xmax/(nx-1)
dz = 2*zmax/(nz-1)
hb = 63.5078 #("AtomicMassUnit" ("Micrometers")^2)/("Milliseconds")
m3 = 3   # AtomicMassUnit
m4 = 4 

pxmax = 2*pi*hb/dx/2
pzmax = 2*pi*hb/dz/2
# pxmax= (nx+1)/2 * 2*pi/(2*xmax)*hb # want this to be greater than p
# pzmax= (nz+1)/2 * 2*pi/(2*zmax)*hb

s = f"""nx = {nx}
nz = {nz} 
xmax = {xmax}
zmax = {zmax}
dt = {dt}
dx = {dx}
dz = {dz}
hb = {hb}
m3 = {m3}
m4 = {m4}
pxmax = {pxmax}
pymax = {pzmax}
"""
l.info(s)
```

```python
l.info(f"""rotate phase per dt for m3 = {1j*hb*dt/(2*m3*dx*dz)} \t #want this to be small
rotate phase per dt for m4 = {1j*hb*dt/(2*m4*dx*dz)} 
number of grid points = {round(nx*nz/1000/1000,3)} (million)
minutes per grid op = {round((nx*nz)*0.001*0.001/60, 3)} \t(for 1μs/element_op)
""")
```

```python
wavelength = 1.083 #Micrometers
beam_angle = 90
k = sin(beam_angle*pi/180/2) * 2*pi / wavelength # effective wavelength
klab = k

#alternatively 
# k = pi / (4*dx)
Nk = 4
k = pi / (Nk*dx)
beam_angle = np.arcsin(k/(2*pi/wavelength))*180/pi

kx = 0
kz = k

# print("k =", k, " is 45 degree Bragg")
# k = 2*pi / wavelength
# k = pi / (2*dz)
# k = pi / (4*dx)
# dopd = 60.1025 # 1/ms Doppler detuning (?)

p = hb*k
# print("k  =",k,"1/µm")
# print("p  =",p, "u*µm/ms")
v4 = 2*hb*k/m4
v3 = 2*hb*k/m3
# print("v3 =",v3, "µm/ms")
# print("v4 =",v4, "µm/ms")

# sanity check
# assert (pxmax > p*2.5 or pzmax > p*2.5), "momentum resolution too small"
# dopd = 60.1025 # 1/ms Doppler detuning (?)
dopd = v3**2 * m3 / hb

l.info(f"""wavelength = {wavelength} µm
beam_angle = {beam_angle}
k = {k} 1/µm
klab = {klab} 
kx = {kx} 1/µm
kz = {kz} 1/µm
p = {p} u*µm/ms
pxmax/p = {pxmax/p} 
pzmax/p = {pzmax/p} 
2p = {2*p} u*µm/ms
v3 = {v3} µm/ms
v4 = {v4} µm/ms
dopd = {dopd}
2*pi/(2*k)/dx = {2*pi/(2*k)/dx} this should be larger than 4 (grids) and bigger the better
""")
if not (pxmax > p*2.5): l.warning(f"p={p} not << pmax={pxmax} momentum resolution too small!")
if not 2*pi/(2*k)/dx > 1:  l.warning(f"2*pi/(2*k)/dx = {2*pi/(2*k)/dx} aliasing will happen")

l.info(f"""xmax/v3 = {xmax/v3} ms is the time to reach boundary
zmax/v3 = {zmax/v3}
""")
```

```python
# V00 = 50000
# dt=0.01
# VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * V00 * cosGrid )
dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb
pxlin = np.linspace(-pxmax,+pxmax,nx)
pzlin = np.linspace(-pzmax,+pzmax,nz)
xlin = np.linspace(-xmax,+xmax, nx)
zlin = np.linspace(-zmax,+zmax, nz)
# print("(dpx,dpz) = ", (dpx, dpz))
if abs(dpx - (pxlin[1]-pxlin[0])) > 0.0001: l.error("AHHHHH px is messed up (?!)")
if abs(dpz - (pzlin[1]-pzlin[0])) > 0.0001: l.error("AHHHHH pz")
if abs(dx - (xlin[1]-xlin[0])) > 0.0001: l.error("AHHHHx")
if abs(dz - (zlin[1]-zlin[0])) > 0.0001: l.error("AHHHHz")
l.info(f"""dpx = {dpx} uµm/m
dpz = {dpz} """)
```

```python
#### WARNING:
###  These frequencies are in Hz, 
#### This simulation uses time in ms, 1Hz = 0.001 /ms
a4 = 0.007512 # scattering length µm
intensity1 = 1 # mW/mm^2 of beam 1
intensity2 = intensity1
intenSat = 0.0017 # mW/mm^2
linewidth = 2*pi*1.6e6 # rad * Hz
omega1 = linewidth * sqrt(intensity1/intenSat/2)
omega2 = linewidth * sqrt(intensity2/intenSat/2)
detuning = 2*pi*3e9 # rad*Hz
omegaRabi = omega1*omega2/2/detuning # rad/s

VR = 2*hb*(omegaRabi*0.001) # Bragg lattice amplitude # USE THIS ONE! 

omega = 50 # two photon Rabi frequency # https://doi.org/10.1038/s41598-020-78859-1
V0 = 2*hb*omega # Bragg lattice amplitude

# tBraggPi = np.sqrt(2*pi*hb)/V0 
tBraggPi = 2*pi/omegaRabi*1000 
tBraggCenter = tBraggPi * 5
tBraggEnd = tBraggPi * 10

V0F = 50*1000
```

```python jupyter={"source_hidden": true}
l.info(f"""a4 = {a4} µm
intensity1 = {intensity1}  # mW/mm^2 of beam 1
intensity2 = {intensity2}  
intenSat  = {intenSat}  # mW/mm^2 Saturation intensity 
linewidth = {linewidth/1e6} # rad * MHz
omega1 = {omega1/1e6} # rad * MHz 
omega2 = {omega2/1e6}
detuning = {detuning/1e6} # rad * MHz 
omegaRabi = {omegaRabi/1e6} # rad * MHz 
omega = {omega}
VR = {VR}
V0 = {V0}
V0F = {V0F}
tBraggPi = {tBraggPi}
tBraggCenter = {tBraggCenter}
tBraggEnd = {tBraggEnd}
""")
```

```python jupyter={"source_hidden": true}
l.info(f"""hb*k**2/(2*m3) = {hb*k**2/(2*m3)} \t/ms
hb*k**2/(2*m4) = {hb*k**2/(2*m4)}
(hb*k**2/(2*m3))**-1 = {(hb*k**2/(2*m3))**-1} \tms
(hb*k**2/(2*m4))**-1 = {(hb*k**2/(2*m4))**-1}
2*pi*hb*k**2/(2*m3) = {2*pi*hb*k**2/(2*m3)} \t rad/ms
2*pi*hb*k**2/(2*m4) = {2*pi*hb*k**2/(2*m4)}
omegaRabi = {omegaRabi*0.001} \t/ms
tBraggPi = {tBraggPi} ms
""")
```

```python
# def V(t):
#     return V0 * (2*pi)**-0.5 * tBraggPi**-1 * np.exp(-0.5*(t-tBraggCenter)**2 * tBraggPi**-2)

# def VB(t, tauMid, tauPi):
#     return V0 * (2*pi)**-0.5 * tauPi**-1 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)

# V0F = 50*1000
# def VBF(t, tauMid, tauPi, V0FArg=V0F):
#     return V0FArg * (2*pi)**-0.5 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)
@njit
def VS(ttt, mid, wid, V0=VR):
    return V0 * 0.5 * (1 + np.cos(2*np.pi/wid*(ttt-mid))) * \
            (-0.5*wid+mid<ttt) * (ttt<0.5*wid+mid)
```

```python
l.warning(f"""2D grid RAM: {(nx*nz)*(np.cdouble(1).nbytes)/1000/1000} MB
4D grid RAM: {(nx*nz)**2*(np.cdouble(1).nbytes)/1000/1000} MB""")
```

```python
psi=np.zeros((nx,nz, nx,nz),dtype=np.cdouble)
l.info(f"psi RAM usage: {round(psi.nbytes/1000/1000 ,3)} MB")
```

```python
process_py = psutil.Process(os.getpid())
def ram_py(): return process_py.memory_info().rss;
def ram_py_MB(): return (ram_py()/1000**2)
def ram_py_GB(): return (ram_py()/1000**3)
def ram_py_log(): l.info(str(round(ram_py()/1000**2,3)) + "MB of system memory used")
ram_sys_MB = psutil.virtual_memory().total/1e6
ram_sys_GB = psutil.virtual_memory().total/1e9
```

```python
l.info(f"Current RAM usage: {round(ram_py_MB(),3)} MB")
```

```python
# xgrid = np.tensordot(xlin, np.ones(nz, dtype=dtyper), axes=0)
# cosXGrid = np.cos(2*k*xgrid)
zgrid = np.tensordot(np.ones(nz),zlin, axes=0)
cosZGrid = np.cos(2*k*zgrid)
l.info(f"cosZGrid size {round(cosZGrid.nbytes/1000**2,3)} MB")
```

```python
# Parameters from scan 20240506-184909-TFF
V3pi1 = 0.08*VR # ratio corrent when using intensity=1
t3pi1 = 49.8e-3 # ms 
e3pi1 = 0.9950  # expected efficiency from scan
V3pi2 = 0.04*VR
t3pi2 = 49.9e-3
e3pi2 = (0.4994, 0.4990)
# 20240507-212137-TFF
V4pi1 = 0.06*VR # ratio corrent when using intensity=1
t4pi1 = 66.4e-3 # ms 
e4pi1 = 0.9950  # expected efficiency from scan
V4pi2 = 0.03*VR
t4pi2 = 66.5e-3
e4pi2 = (0.4990, 0.4994)
# 20240509-181745-TFF 
V4sc = 0.135*VR
t4sc = 22.8e-3 
e4sc = 0.4238 
```

```python
l.info(f"""(V3pi1, t3pi1, e3pi1) = {(V3pi1, t3pi1, e3pi1)}
(V3pi2, t3pi2, e3pi2) = {(V3pi2, t3pi2, e3pi2)}
(V4pi1, t4pi1, e4pi1) = {(V4pi1, t4pi1, e4pi1)}
(V4pi2, t4pi2, e4pi2) = {(V4pi2, t4pi2, e4pi2)}
(V4sc,  t4sc,  e4sc ) = {(V4sc,  t4sc,  e4sc)}""")
```

```python
if (intensity1 != 1) or (intensity2 != 1): l.warning(f"{intensity1}, {intensity2} should be 1, or some code got messed up (?)")
```

```python
tbtest = np.arange(0, max([t3pi1,t3pi2,t4pi1,t4pi2,t4sc]),dt)
plt.figure()
plt.plot(tbtest, VS(tbtest, 0.5*t3pi1, t3pi1, V3pi1),label="$\pi$    pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t3pi2, t3pi2, V3pi2),label="$\pi/2$ pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4pi1, t4pi1, V4pi1),label="$\pi$    pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4pi2, t4pi2, V4pi2),label="$\pi/2$ pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4sc,  t4sc,  V4sc),label="scat pulse for ${}^4\mathrm{He}$")
plt.legend()
plt.xlabel("t (ms)")
plt.ylabel("VS")
plt.show()
```

```python
# vtest = np.cos(2*k*xlin)
ncrop = int(0.3*nx)
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
# plt.imshow(cosXGrid.T)
plt.imshow(cosZGrid.T)
plt.title("bragg potential grid smooth?")

plt.subplot(2,2,2)
# plt.imshow(cosXGrid[:ncrop,:ncrop].T)
plt.imshow(cosZGrid[:ncrop,:ncrop].T)
plt.title("grid zoomed in")

plt.subplot(2,2,3)
# plt.plot(cosXGrid[:,0],alpha=0.9,linewidth=0.5)
plt.plot(cosZGrid[0,:],alpha=0.9,linewidth=0.5)

plt.subplot(2,2,4)
# plt.plot(cosXGrid[:ncrop,0],alpha=0.9,linewidth=0.5)
plt.plot(cosZGrid[0,:ncrop],alpha=0.9,linewidth=0.5)

title="bragg_potential_grid"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show()
```

```python

```

```python
gc.collect()
%reset -f in
%reset -f out
```

```python

```

## Initial wave function

```python
sg = 8;
@njit(cache=True)
def psi0gaussianNN(x3, z3, x4, z4, sx3=sg, sz3=sg, sx4=sg, sz4=sg, px3=0.0, pz3=0.0, px4=0.0, pz4=0.0):
    return    np.exp(-0.5*x3**2/sx3**2)\
            * np.exp(-0.5*z3**2/sz3**2)\
            * np.exp(-0.5*x4**2/sx4**2)\
            * np.exp(-0.5*z4**2/sz4**2)\
            * np.exp(+(1j/hb)*(px3*x3 + pz3*z3 + px4*x4 + pz4*z4))

@njit(cache=True)
def check_norm(psi,dx=dx,dz=dz) -> dtyper:
    return np.trapz(np.trapz(np.trapz(np.trapz(np.abs(psi)**2))))*(dx*dx*dz*dz)


@njit(parallel=True, cache=True)
def psi0gaussian(sx3=sg, sz3=sg, sx4=sg, sz4=sg, px3=0, pz3=0, px4=0, pz4=0, xlin=xlin,zlin=zlin) -> np.ndarray:
    psi=np.zeros((nx,nz, nx,nz),dtype=dtypec)
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin, z3, x4, z4,sx3,sz3,sx4,sz4,px3,pz3,px4,pz4)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtypec)

# @njit(cache=True)
# @njit()
# @jit
@jit(forceobj=True, cache=True)
def only3(psi):
    return np.trapz(np.trapz(np.abs(psi)**2 ,dx=dz,axis=3),dx=dx,axis=2)

# @njit(cache=True)
# @njit()
# @jit
@jit(forceobj=True, cache=True)
def only4(psi):
    return np.trapz(np.trapz(np.abs(psi)**2 ,dx=dx,axis=0),dx=dz,axis=0)
```

```python
_ = psi0gaussian() # executes in 8.97s 
```

```python
@njit(parallel=True, cache=True)
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
    
    psi = np.zeros((nx,nz,nx,nz),dtype=dtypec)
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0gaussianNN(xlin-dx3,z3-dz3, x4+dx4,z4+dz4, s3,s3, s4,s4,+px,+pz,-px,-pz)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtypec)
    
```

```python
su=5
_ = psi0_just_opposite(dr=0,s3=su,s4=su,pt=-3*hb*k,a=0*pi)
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

@njit(cache=True, parallel=True)
def psi0Pair(dr=20,s3=sg,s4=sg,pt=p,a=0,nx=nx,nz=nz):
    psi = np.zeros((nx,nz,nx,nz),dtype=dtypec)
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                psi[:,iz3,ix4,iz4] = psi0PairNN(xlin, z3, x4, z4, dr,s3,s4,pt,a)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtypec)
```

```python
_ = psi0Pair() # 5.24s 
```

```python
@njit(nogil=True,parallel=True,cache=True) # psi0Pair already parallelised
def psi0ring_loop_helper(dr,s3,s4,pt,an, 
#                         ):
                         progress_proxy=None):
    psi = np.zeros((nx,nz,nx,nz),dtype=dtypec)
    angles_list = np.linspace(0,pi,an+1)[:-1]
    for ia in range(an):
        a = angles_list[ia]
        psi += (1/an) * psi0Pair(dr=dr,s3=s3,s4=s4,pt=pt,a=a,nx=nx,nz=nz)
        if progress_proxy != None:
            progress_proxy.update(1)
    return psi.astype(dtypec)

# @njit(cache=True)
def psi0ring_with_logging(dr=20,s3=3,s4=3,pt=p,an=4):   
    angles_list = np.linspace(0,pi,an+1)[:-1]
    print("angles_list = "+ str(np.round(angles_list/pi,4)) + " * pi")
    with ProgressBar(total=an) as progress:
        psi = psi0ring_loop_helper(dr,s3,s4,pt,an,progress)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtypec)
    
@njit(parallel=True,cache=True)
def psi0ring(dr=20,s3=3,s4=3,pt=p,an=4):   
    psi = psi0ring_loop_helper(dr,s3,s4,pt,an)
    normalisation = check_norm(psi)
    return (psi/sqrt(normalisation)).astype(dtypec)
```

```python
# _ = psi0ring_with_logging(dr=20,s3=3,s4=3,pt=p,an=1)
_ = psi0ring(dr=20,s3=3,s4=3,pt=p,an=1) # 13s
```

```python
gc.collect()
%reset -f in
%reset -f out
```

<!-- #raw -->
sys.getsizeof(psi)
<!-- #endraw -->

```python

```

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

```

```python
su = 3
# psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=-100, pz3=-50, px4=100, pz4=50)
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)
t = 0
```

```python
tempTest3 = only3(psi)
tempTest4 = only4(psi)
print("check normalisation psi", check_norm(psi))
phi, swnf = phiAndSWNF(psi)
tempTest3phi = only3phi(phi)
tempTest4phi = only4phi(phi)
print("check normalisation phi", check_norm_phi(phi))
print("swnf =", swnf)

plt.figure(figsize=(10,6))
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
ram_py_log()
del tempTest3, tempTest4, tempTest3phi, tempTest4phi
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
```

```python

```

```python

```

<!-- #raw -->
su = 4
t = 0
psi = psi0ring_with_logging(dr=5,s3=1.7,s4=1.7,pt=2*hb*k,an=128) # Takes about 12m
with pgzip.open(output_prefix+'psi0ring_with_logging_testRun_128'+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 
<!-- #endraw -->

<!-- #raw -->
with pgzip.open('/Volumes/tonyNVME Gold/twoParticleSim/20240507-191333-TFF/psi0ring_with_logging_testRun_128.pgz.pkl'
                , 'rb', thread=8) as file:
    psi = pickle.load(file)
su, t = 4, 0 
<!-- #endraw -->

```python

```

<!-- #raw -->
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
ram_py_log()
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
<!-- #endraw -->

```python
_ = None
print_ram_usage(globals().items(),10)
```

```python
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

```python

```

```python

```

```python
# @njit(cache=True, parallel=True)
@jit(forceobj=True, parallel=True)
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
    
    plt.imshow(np.flipud(gx3x4.T),extent=[-xmax,xmax,-xmax,xmax],cmap='Greens')
    plt.title("$g^{(2)}_{\pm\pm}$ of $z_\mathrm{cut} = "+str(cut)+"dz$ and $E="+str(round(E,4))+"$")
    plt.xlabel("$x_3$")
    plt.ylabel("$x_4$")
    plt.axhline(y=0,color='k',alpha=0.2,linewidth=0.7)
    plt.axvline(x=0,color='k',alpha=0.2,linewidth=0.7)
    plt.text(+xmax*0.6,+xmax*0.8,"$g^{(2)}_{++}="+str(round(gpp,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-xmax*0.6,+xmax*0.8,"$g^{(2)}_{-+}="+str(round(gmp,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(+xmax*0.6,-xmax*0.8,"$g^{(2)}_{+-}="+str(round(gpm,4))+"$", color='white',ha='center',alpha=0.9)
    plt.text(-xmax*0.6,-xmax*0.8,"$g^{(2)}_{--}="+str(round(gmm,4))+"$", color='white',ha='center',alpha=0.9)
    
```

```python
plt.figure(figsize=(14,6))

cut_list = [1.0, 10.0, 15.0]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gx3x4_calc(phi,cut=cut)
    plt.subplot(1,3,i+1)
    plot_gx3x4(gx3x4,cut)
plt.show()
```

```python
p/dpz
```

```python

```

```python

```

```python

```

```python

```

## Scattering

```python
# numba.set_num_threads(2)
# Each thread will multiply the memory usage!
# e.g. single thread 10GB -> 2 threads ~30GB
```

```python
l.info(f"""numba.get_num_threads() = {numba.get_num_threads()}, 
{round(ram_py_GB(),2)} GB used of sys total {round(ram_sys_GB,2)} GB in system
multithread could use up to {round(ram_py_GB()*numba.get_num_threads(),2)} GB !!!""")
```

```python

```

```python
# TODO WTF did this come from?
# a34 = 0.029 #µm
# a34 = 0.5 #µm
a34 = sqrt(2)*5*dx
strength34 = 1e5 # I don't know
l.info(f"{-(1j/hb) * strength34 * np.exp(-((0-0)**2 +(0-0)**2)/(4*a34**2)) *0.5*dt}")
l.info(f"a34 = {a34}")
```

<!-- #raw -->
expContact = np.zeros((nx,nz, nx,nz),dtype=dtypec)
for (iz3, z3) in enumerate(zlin):
    for (ix4, x4) in enumerate(xlin):
        for (iz4, z4) in enumerate(zlin):
            dis = ((xlin-x4)**2 +(z3-z4)**2)**0.5
            expContact[:,iz3,ix4,iz4] = np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        strength34 *
                                            0.5*(1+np.cos(2*np.pi/a34*( dis ))) * 
                                            (-0.5*a34 < dis) * (dis < 0.5*a34)
                                        # np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
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
plt.imshow(np.flipud(np.angle(expContact[50,:,53,:]).T))
plt.colorbar()
plt.show()
<!-- #endraw -->

<!-- #raw -->
expContact = None
del expContact
gc.collect()
<!-- #endraw -->

```python

```

```python
@njit(parallel=True,cache=True)
# @njit
def scattering_evolve_loop_helper2_inner_psi_step(psi_init, s34=strength34):
    psi = psi_init
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                dis = ((xlin-x4)**2 +(z3-z4)**2)**0.5
                psi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * # this one is unitary time evolution operator
                                        s34 *
                                             0.5*(1+np.cos(2*np.pi/a34*( dis ))) * 
                                            (-0.5*a34 < dis) * (dis < 0.5*a34)
                                        # np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
                                               # inside the guassian contact potential
                                               *0.5*dt
                                            )
    return psi

@njit(parallel=True,cache=True)
# @njit
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
def scattering_evolve_loop_helper2(t_init, psi_init, swnf, steps=20, progress_proxy=None, s34=strength34):
    t = t_init
    psi = psi_init
    for ia in prange(steps):
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi,s34)
        phi = toPhi(psi, swnf, nthreads=7)
        phi = scattering_evolve_loop_helper2_inner_phi_step(phi)
        #del psi  # might cause memory issues
        psi = toPsi(phi, swnf, nthreads=7)
        psi = scattering_evolve_loop_helper2_inner_psi_step(psi,s34)
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)                                   
    return (t, psi, phi)
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

#TODO: WTF WAS THIS!!!!!????!?!?!??!?!?!? 
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
    
    psi = np.zeros((nx,nz,nx,nz),dtype=dtypec)
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
    return (psi/sqrt(normalisation)).astype(dtypec)
```

```python
output_pre_selp = output_prefix + "scattering_evolve_loop_plot/"
os.makedirs(output_pre_selp, exist_ok=True)
def scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(ram_py_MB(),3)) + "MB  ")

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
        title= "f="+str(f)+",t="+t_str 
        plt.savefig(output_pre_selp+title+".pdf", dpi=600)
        plt.savefig(output_pre_selp+title+".png", dpi=600)
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 
```

```python

```

```python
output_pre_selpa = output_prefix + "scattering_evolve_loop_plot_alt/"
os.makedirs(output_pre_selpa, exist_ok=True)
def scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=1,po=1):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(ram_py_MB(),3)) + "MB  ")

    fig = plt.figure(figsize=(12,7))
    plt.subplot(2,2,1)
#     plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
#     plt.imshow(np.emath.logn(power,np.flipud(only3(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
#     plt.imshow(np.power(np.flipud(only3(psi).T),power), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
    plt.imshow(np.power(np.log(logPlus+np.flipud(only3(psi).T)),po), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")
    plt.title("$t="+t_str+" \ ms $")

    plt.subplot(2,2,2)
#     plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
#     plt.imshow(np.power(np.flipud(only4(psi).T),power), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
#     plt.imshow(np.emath.logn(power,np.flipud(only4(psi).T)), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
    plt.imshow(np.power(np.log(logPlus+np.flipud(only4(psi).T)),po), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
    plt.xlabel("$x \ (\mu m)$")
    plt.ylabel("$z \ (\mu m)$")

    plt.subplot(2,2,3)
#     plt.imshow((only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
#     plt.imshow(np.power(only3phi(phi).T,power), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
#     plt.imshow(np.emath.logn(power,only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
    # plt.colorbar()
    plt.imshow(np.power(np.log(logPlus+only3phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.ylabel("$p_z \ (\hbar k)$")

    plt.subplot(2,2,4)
#     plt.imshow((only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
#     plt.imshow(np.power(only4phi(phi).T,power), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
#     plt.imshow(np.emath.logn(power,only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
    plt.imshow(np.power(np.log(logPlus+only4phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
    # plt.colorbar()
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.xlabel("$p_x \ (\hbar k)$")

    if plt_save:
        title= "f="+str(f)+",t="+t_str+",logPlus="+str(logPlus) 
        plt.savefig(output_pre_selpa+title+".pdf", dpi=600)
        plt.savefig(output_pre_selpa+title+".png", dpi=600)
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 
```

```python
su = 3
# psi = psi0_just_opposite_double(dr=0,s3=su*(4/3),s4=su,pt=-4.0*hb*k,a=0.5*pi) # 16.37s
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)
t = 0
f = 0
phi, swnf = phiAndSWNF(psi, nthreads=7)
```

```python
scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False)
```

```python

```

```python

```

<!-- #raw -->
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=10)
<!-- #endraw -->

```python
_ = None
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
```

```python
evolve_s34 = 1e6
l.info(f"strength34 = {strength34}, \t evolve_s34 = {evolve_s34}")
```

```python

```

```python

```

```python
numba.get_num_threads()
```

# Simulation Sequence ??? (dev)

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Scattering from perfect init state
<!-- #endregion -->

```python
evolve_loop_time_start = datetime.now()
_ = scattering_evolve_loop_helper2(t,psi,swnf,steps=1,progress_proxy=None,s34=strength34)
evolve_loop_time_end = datetime.now()
evolve_loop_time_delta = evolve_loop_time_end - evolve_loop_time_start
l.info(f"Time to run one evolve_loop_time_start is {evolve_loop_time_delta} (Run again to use cached compile)")
# need to run this once before looping to cache numba compiles
```

```python
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
```

```python
print_every = 10
frames_count = 500
total_steps = print_every * frames_count
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""print_every = {print_every}, \tframes_count = {frames_count}, total_steps = {total_steps}
Target simulation end time = {frames_count*print_every*dt} ms
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")
```

```python
assert False, "just to catch run all"
```

```python
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
evolve_many_loops_start = datetime.now()
loopAccum = 0 
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        evolve_many_loops_inner_now = datetime.now()
        tE = evolve_many_loops_inner_now - evolve_many_loops_start
        if loopAccum > 0: 
            tR = (frames_count-loopAccum)* tE/loopAccum
            l.info(f"Now l={loopAccum}, t={round(t,6)}, tE={tE}, tR={tR}")
        scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
        scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        gc.collect()
        (t,psi,phi) = scattering_evolve_loop_helper2(t,psi,swnf,
                                                     steps=print_every,progress_proxy=progressbar,s34=evolve_s34)
        loopAccum += 1
f += 1
scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
```

```python
l.info(t)
with pgzip.open(output_prefix+f"psi at t={t}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 
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

## Scattering generated from Bragg pulse

```python
@njit(parallel=True,cache=True)
def scattering_evolve_bragg_loop_helper2_inner_psi_step(
        psi_init, s34, tnow,
        t3mid, t3wid, v3pot,
        t4mid, t4wid, v4pot,
    ):
    psi = psi_init
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                # xlin is x3 here in the for loop
                # a34 contact potential
                dis = ((xlin-x4)**2 +(z3-z4)**2)**0.5
                psi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb)*0.5*dt* # this one is unitary time evolution operator
                                             s34 * 0.5*(1+np.cos(2*np.pi/a34*( dis ))) * 
                                                 (-0.5*a34 < dis) * (dis < 0.5*a34)
                                        # np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2)) # inside the guassian contact potential
                                            )
                # Bragg Potential (VxExpGrid in oneParticle)
                VS3 = VS(tnow,t3mid,t3wid,v3pot)
                VS4 = VS(tnow,t4mid,t4wid,v4pot)
                if VS3 != 0 or VS4 != 0:
                    psi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb)*0.5*dt*(
                                             VS3*np.cos(2*kx*xlin + 2*kz*z3) + 
                                             VS4*np.cos(2*kx*x4   + 2*kz*z4)
                                            ))
                else: continue
                #
    return psi

# @njit(parallel=True,cache=True)
# def scattering_evolve_bragg_loop_helper2_inner_phi_step(phi_init):
#     phi = phi_init
#     for iz3 in prange(nz):
#         pz3 = pzlin[iz3]
# #     for (iz3, pz3) in enumerate(pzlin):
#         for (ix4, px4) in enumerate(pxlin):
#             for (iz4, pz4) in enumerate(pzlin):
#                 phi[:,iz3,ix4,iz4] *= np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin**2 + pz3**2) \
#                                              -(1j/hb) * (0.5/m4) * (dt) * (  px4**2 + pz4**2))
#     return phi

# @jit(nogil=True, parallel=True, forceobj=True)
@jit(nogil=True, forceobj=True)
# @njit(nogil=True, parallel=True)
def scattering_evolve_bragg_loop_helper2(
        tin, psii, swnf, 
        steps=20, progress_proxy=None, s34=strength34,
        t3mid=-2, t3wid=1, v3pot=0,
        t4mid=-2, t4wid=1, v4pot=0,
    ):
    t = tin
    psi = psii
    for ia in range(steps):
        psi = scattering_evolve_bragg_loop_helper2_inner_psi_step(psi,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)
        phi = toPhi(psi, swnf, nthreads=7)
        phi = scattering_evolve_loop_helper2_inner_phi_step(phi)
        #del psi  # might cause memory issues
        psi = toPsi(phi, swnf, nthreads=7)
        psi = scattering_evolve_bragg_loop_helper2_inner_psi_step(psi,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)                                   
    return (t, psi, phi)
```

```python

```

```python
evolve_loop_time_start = datetime.now()
_ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,steps=1,progress_proxy=None,s34=evolve_s34,
                                        t3mid=-3,t3wid=1,v3pot=0,
                                        t4mid=0.5*t4sc,t4wid=t4sc,v4pot=V4sc)
evolve_loop_time_end = datetime.now()
evolve_loop_time_delta = evolve_loop_time_end - evolve_loop_time_start
l.info(f"Time to run one evolve_loop_time_start is {evolve_loop_time_delta} (Run again to use cached compile)")
# need to run this once before looping to cache numba compiles
```

```python
VS(1*dt ,0.5*t4sc, t4sc, V4sc)
```

```python
VS(10*dt ,-3,1, V4sc)
```

```python
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
print_ram_usage(globals().items(),10)
```

```python
print_every = 10
frames_count = 500
total_steps = print_every * frames_count
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""print_every = {print_every}, \tframes_count = {frames_count}, total_steps = {total_steps}
Target simulation end time = {frames_count*print_every*dt} ms
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")
```

```python
assert False, "just to catch run all"
```

```python
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
evolve_many_loops_start = datetime.now()
loopAccum = 0 
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        evolve_many_loops_inner_now = datetime.now()
        tP = evolve_many_loops_inner_now - evolve_many_loops_start
        if loopAccum > 0: 
            tR = (frames_count-loopAccum)* tP/loopAccum
            tE = datetime.now() + tR
            l.info(f"Now l={loopAccum}, t={round(t,6)}, tP={tP}, tR={tR}, tE = {tE}")
        scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
        scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        gc.collect()
        (t,psi,phi) = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                          steps=print_every,progress_proxy=progressbar,s34=evolve_s34,
                          t3mid=-3, t3wid=1, v3pot=0, 
                          t4mid=0.5*t4sc, t4wid=t4sc, v4pot=V4sc                        
                          )
        loopAccum += 1
f += 1
scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
```

```python
l.info(t)
with pgzip.open(output_prefix+f"psi at t={t}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 
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

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Exporting to Video (Quite high VM RAM usage)
<!-- #endregion -->

```python
assert False, "just to catch run all"
```

```python
# Function to extract frame number from filename
def extract_frame_number(filename):
    match = re.search(r'f=(\d+)', filename)
    return int(match.group(1)) if match else None
img_alt_list = glob.glob(output_pre_selpa+'*.png')
img_alt_list.sort(key=lambda x: extract_frame_number(x))
```

```python
N_JOBS
```

```python
# img_alt_frames = []  # Read and process images, storing them in a list
# for image in tqdm(img_alt_list, desc="Processing Images"):
#     img = cv2.imread(image)
#     img_alt_frames.append(img)  # Append processed frame to list
img_alt_frames = Parallel(n_jobs=-1)(
    delayed(lambda image: cv2.imread(image))(image) for image in tqdm(img_alt_list, desc="Processing Images")
)
```

```python
if img_alt_frames:  # Determine the width and height from the first image if not empty
    height, width, layers = img_alt_frames[0].shape # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'hvc1')  # HEVC codec
    out = cv2.VideoWriter(output_prefix+"scattering_evolve_loop_plot_alt.mov", 
                          fourcc, 10.0, (width, height), True) # Write img_alt_frames to the video file
    for frame in tqdm(img_alt_frames, desc="Writing Video"):
        out.write(frame)  # Write out frame to video
    out.release()  # Release the video writer
    del frame, out
else:
    print("No images found or processed.")
del img_alt_frames, frame
gc.collect()
```

<!-- #raw -->
# Create a clip from the images sequence
clip = ImageSequenceClip(img_alt_list, fps=10)  # fps can be adjusted
# Write the clip to a video file in MOV format
clip.write_videofile(output_prefix+"scattering_evolve_loop_plot_alt.mp4",threads=8,fps=10.0)
# Close the clip to free resources
clip.close()
<!-- #endraw -->

<!-- #raw -->
# Determine the width and height from the first image
frame = cv2.imread(img_alt_list[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'hvc1')  # Be sure to use lower case
out = cv2.VideoWriter(output_prefix+"scattering_evolve_loop_plot_alt.mov", fourcc, 10.0, (width, height))

for image in tqdm(img_alt_list):
    img = cv2.imread(image)
    out.write(img)  # Write out frame to video

out.release()  # Release the video writer
<!-- #endraw -->

```python
gc.collect()
%reset -f in
%reset -f out
ram_py_log()
print_ram_usage(globals().items(),10)
```

```python

```

```python

```

```python

```

```python

```
