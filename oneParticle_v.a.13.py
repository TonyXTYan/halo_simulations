# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] editable=true slideshow={"slide_type": ""}
# # One Particle
# -

# ## Setup

# !python -V

# + editable=true jupyter={"is_executing": true} slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
import numpy as np
from math import *
from uncertainties import *
from scipy.stats import chi2
import scipy
from matplotlib import gridspec
import matplotlib
from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
cm = 1/2.56

import pandas as pd
import sys
import statsmodels.api as sm
import warnings ## statsmodels.api is too old ... -_-#

import pickle
import pgzip
import os
import platform
import logging
import re
import cv2
# import sys

from joblib import Parallel, delayed
import joblib

from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import time

import pyfftw


# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots

# +
# from numba import njit, jit, prange, objmode, vectorize
# import numba
# numba.set_num_threads(8)
# -

N_JOBS=8
N_JOB2=5
nthreads=2

import gc
gc.enable()

# +
use_cache = False
save_cache = False
save_debug = True 

datetime_init = datetime.now()

# os.makedirs('output/oneParticleSim', exist_ok=True)
output_prefix = "output/oneParticleSim/"+\
                datetime_init.strftime("%Y%m%d-%H%M%S") + "-" + \
                str("T" if save_debug else "F") + \
                str("T" if use_cache else "F") + \
                str("T" if save_cache else "F") + \
                "/"
output_ext = ".pgz.pkl"
os.makedirs(output_prefix, exist_ok=True)
print(output_prefix)
# -

plt.set_loglevel("warning")

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

l.info(f"This file is {output_prefix}logs.log")

#

l.info(f"""nthreads = {nthreads}
N_JOBS = {N_JOBS}
N_JOB2 = {N_JOB2}""")


# +
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


# -

print_ram_usage()

# +
#test 4

# +
# np.show_config()

# +
nx = 1500+1
nz = 1500+1
xmax = 50 #Micrometers
# zmax = (nz/nx)*xmax
zmax = xmax
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
# -

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

# + editable=true slideshow={"slide_type": ""}
l.info(f"""rotate phase per dt for m3 = {1j*hb*dt/(2*m3*dx*dz)} \t #want this to be small
rotate phase per dt for m4 = {1j*hb*dt/(2*m4*dx*dz)} 
number of grid points = {round(nx*nz/1000/1000,3)} (million)
minutes per grid op = {round((nx*nz)*0.001*0.001/60, 3)} \t(for 1μs/element_op)
""")

# + editable=true slideshow={"slide_type": ""}
wavelength = 1.083 #Micrometers
beam_angle = 90
k = sin(beam_angle*pi/180/2) * 2*pi / wavelength # effective wavelength
klab = k

#alternatively 
# k = pi / (4*dx)
# Nk = 4
# k = pi / (Nk*dx)
# beam_angle = np.arcsin(k/(2*pi/wavelength))*180/pi

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
# dopd = v3**2 * m3 / hb
dopd = v4**2 * m4 / hb /2
# -

xmax**2 * m3 * 6 / (hb*pi*(nx-1))

# + editable=true slideshow={"slide_type": ""}
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
dopd = {dopd} 1/ms
2*pi/(2*k)/dx = {2*pi/(2*k)/dx} this should be larger than 4 (grids) and bigger the better
""")
if not (pxmax > p*2.5): l.warning(f"p={p} not << pmax={pxmax} momentum resolution too small!")
if not 2*pi/(2*k)/dx > 1:  l.warning(f"2*pi/(2*k)/dx = {2*pi/(2*k)/dx} aliasing will happen")
# -

hb*pi*(nx-1) / (2*m3*xmax*6)

# + editable=true slideshow={"slide_type": ""}
l.info(f"""xmax/v3 = {xmax/v3} ms is the time to reach boundary
zmax/v3 = {zmax/v3}
""")
# -

# V00 = 50000
# dt=0.01
# VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * V00 * cosGrid )
dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb
pxlin = np.linspace(-pxmax,+pxmax,nx)
pzlin = np.linspace(-pzmax,+pzmax,nz)
# print("(dpx,dpz) = ", (dpx, dpz))
if abs(dpx - (pxlin[1]-pxlin[0])) > 0.0001: l.error("AHHHHH px is messed up (?!)")
if abs(dpz - (pzlin[1]-pzlin[0])) > 0.0001: l.error("AHHHHH pz")
l.info(f"""dpx = {dpx} uµm/m
dpz = {dpz} """)

# + editable=true slideshow={"slide_type": ""}


# +
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

# + editable=true slideshow={"slide_type": ""}
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
tBraggPi = {tBraggPi} ms
tBraggCenter = {tBraggCenter} ms
tBraggEnd = {tBraggEnd} ms
""")
# -

l.info(f"""hb*k**2/(2*m3) = {hb*k**2/(2*m3)} \t/ms
hb*k**2/(2*m4) = {hb*k**2/(2*m4)}
(hb*k**2/(2*m3))**-1 = {(hb*k**2/(2*m3))**-1} \tms
(hb*k**2/(2*m4))**-1 = {(hb*k**2/(2*m4))**-1}
2*pi*hb*k**2/(2*m3) = {2*pi*hb*k**2/(2*m3)} \t rad/ms
2*pi*hb*k**2/(2*m4) = {2*pi*hb*k**2/(2*m4)}
omegaRabi = {omegaRabi*0.001} \t/ms
tBraggPi = {tBraggPi} ms
""")


# + editable=true slideshow={"slide_type": ""}
def V(t):
    return V0 * (2*pi)**-0.5 * tBraggPi**-1 * np.exp(-0.5*(t-tBraggCenter)**2 * tBraggPi**-2)

def VB(t, tauMid, tauPi):
    return V0 * (2*pi)**-0.5 * tauPi**-1 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)

V0F = 50*1000
def VBF(t, tauMid, tauPi, V0FArg=V0F):
    return V0FArg * (2*pi)**-0.5 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)


# + editable=true slideshow={"slide_type": ""}
l.info(f"term infront of Bragg potential {1j*(dt/hb)}")
l.info(f"max(V) {1j*(dt/hb)*V(tBraggCenter)}")


# -

# @njit(cache=True)
def VS(ttt, mid, wid, V0=VR):
    return V0 * 0.5 * (1 + np.cos(2*np.pi/wid*(ttt-mid))) * \
            (-0.5*wid+mid<ttt) * (ttt<0.5*wid+mid)


tbtest = np.arange(tBraggCenter-5*tBraggPi,tBraggCenter+5*tBraggPi,dt)
plt.plot(tbtest, VBF(tbtest,tBraggPi*5,tBraggPi))
plt.plot(tbtest, VS(tbtest,tBraggPi/2,tBraggPi,0.3*V0F))
plt.plot(tbtest, VS(tbtest,tBraggPi/2+0.4,tBraggPi,0.3*VR))
plt.show()
l.info(f"max(V) {1j*(dt/hb)*VBF(tBraggCenter,tBraggPi*5,tBraggPi)}")

xlin = np.linspace(-xmax,+xmax, nx)
zlin = np.linspace(-zmax,+zmax, nz)
psi=np.zeros((nx,nz),dtype=complex)
zones = np.ones(nz)
xgrid = np.tensordot(xlin,zones,axes=0)
# cosGrid = np.cos(2*k*xgrid)
cosGrid = np.cos(2 * kx * xlin[:, np.newaxis] + 2 * kz * zlin)

l.info(f"""2*kz*dx = {2*kz*dx}
dopd*dt {dopd*dt}""")

plt.plot(zlin, np.cos(2*kz*zlin))
plt.plot(zlin, np.cos(2*kz*zlin + dopd*dt*1))
plt.plot(zlin, np.cos(2*kz*zlin + dopd*dt*2))
plt.plot(zlin, np.cos(2*kz*zlin + dopd*dt*3))
plt.xlim(-1,1)
plt.show()

if abs(dx - (xlin[1]-xlin[0])) > 0.0001: l.error("AHHHHx")
if abs(dz - (zlin[1]-zlin[0])) > 0.0001: l.error("AHHHHz")

# + editable=true slideshow={"slide_type": ""}
l.info(f"{round(psi.nbytes/1000/1000 ,3)} MB of data used to store psi")

# + editable=true slideshow={"slide_type": ""}
ncrop = 30
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cosGrid.T,aspect=1)
plt.title("bragg potential grid smooth?")

plt.subplot(2,2,2)
plt.imshow(cosGrid[:ncrop,:ncrop].T,aspect=1)
plt.title("grid zoomed in")

plt.subplot(2,2,3)
plt.plot(cosGrid[0,:],alpha=0.9,linewidth=0.1)

plt.subplot(2,2,4)
plt.plot(xlin[:ncrop],cosGrid[0,:ncrop],alpha=0.9,linewidth=0.5)
plt.plot(xlin[:ncrop],cosGrid[0,1:ncrop+1],alpha=0.9,linewidth=0.5)
plt.plot(xlin[:ncrop],cosGrid[0,10:ncrop+10],alpha=0.9,linewidth=0.5)

title="bragg_potential_grid"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show()


# + editable=true slideshow={"slide_type": ""}
def plot_psi(psi, plt_show=True):
    """Plots $\psi$ of position wavefunction

    Args:
        psi (ndarray 2d): wavefunction dtype=complex
    """    
    
    plt.figure(figsize=(12, 4))
    extent = np.array([-xmax, +xmax, -zmax, +zmax])
    plt.subplot(1, 3, 1)
    plt.imshow(np.flipud(np.abs(psi.T)**2), extent=extent, interpolation='none')
    plt.ylabel("$z$ (µm)")
    plt.xlabel("$x$ (µm)")
    plt.title("Position $|\psi|^2$")
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.flipud(np.real(psi.T)), extent=extent, interpolation='none')
    plt.xlabel("$x$ (µm)")
    plt.title("$\mathrm{Re}(\psi)$")
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.flipud(np.imag(psi.T)), extent=extent, interpolation='none')
    plt.xlabel("$x$ (µm)")
    plt.title("$\mathrm{Im}(\psi)$")
    
    if plt_show: plt.show()


# +
def plot_mom(psi, zoom_div2=15, zoom_div3=6, plt_show=True):
    """Plots momentum wavefunction

    Args:
        psi (ndarray 2d): complex position wavefunction
    """    
    
    
    plt.figure(figsize=(12,3))
    pspace = np.fft.fftfreq(nx)
    extent = np.array([-pxmax,+pxmax,-pzmax,+pzmax])/(hb*k)
#     psifft = np.fft.fftshift(np.fft.fft2(psi))
    psifft = np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))
    psiAbsSqUnNorm = np.abs(psifft)**2
    swnf = sqrt(np.sum(psiAbsSqUnNorm)*dpx*dpz)
    psiAbsSq = psiAbsSqUnNorm / swnf
#     print(np.sum(psiAbsSq)*dpx*dpz)
#     plotdata = np.flipud(psiAbsSq.T)
    plotdata = (psiAbsSq.T)
    
    plt.subplot(1,3,1)
    plt.imshow(plotdata,extent=extent, interpolation='none') 
    plt.ylabel("$p_z \ (\hbar k)$")
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.title("Momentum $|\phi|^2$")
    
    plt.subplot(1,3,2)
    nxm = int((nx-1)/2)
    nzm = int((nz-1)/2)
    nx2 = int((nx-1)/zoom_div2)
    nz2 = int((nz-1)/zoom_div2)
    plotdata = (psiAbsSq[nxm-nx2:nxm+nx2,nzm-nz2:nzm+nz2].T)
    plt.imshow(plotdata,
               extent=np.array([pxlin[nxm-nx2],pxlin[nxm+nx2],pzlin[nzm-nz2],pzlin[nzm+nz2]])/(hb*k),
               interpolation='none') 
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.title("zoomed in")
    
    
    plt.subplot(1,3,3)
#     nx3 = int((nx-1)/200)
#     nz3 = int((nz-1)/200)
#     plotdata = np.flipud(psiAbsSq[nxm-nx3:nxm+nx3,nzm-nz3:nzm+nz3].T)
#     plt.imshow(plotdata,extent=extent*0.01) 
# #     print(nx2,nz2,nx3,nz3)
#     nxm = int((nx-1)/2)
#     nzm = int((nz-1)/2)
    nx2 = int((nx-1)/zoom_div3)
#     nz2 = int((nz-1)/zoom_div3)
    
    plt.plot((pxlin[nxm-nx2:nxm+nx2])/(hb*k), np.trapz(psiAbsSq,axis=1)[nxm-nx2:nxm+nx2])
    plt.axvline(x= 0,color='r',alpha=0.2)
    plt.axvline(x=+2,color='r',alpha=0.2)
    plt.axvline(x=-2,color='r',alpha=0.2)
#     plt.axvline(x=+4,color='r',alpha=0.2)
#     plt.axvline(x=-4,color='r',alpha=0.2)
#     plt.xlabel("$p (u\cdot \mu m/ms)$")
#     plt.ylabel("$|\phi(p)|^2$")
    plt.xlabel("$p_x \ (\hbar k)$")
    plt.title("integrated over $p_z$")
    
#     plt.savefig("output/"+title+".pdf", dpi = 300) 
#     plt.savefig("output/"+title+".png", dpi = 300) 
#     plt.show()
    
    if plt_show: plt.show()
        
# -

sg=0.2
def psi0(x,z,sx=sg,sz=sg,px=0,pz=0):
    return (1/np.sqrt(pi*sx*sz)) \
            * np.exp(-0.5*x**2/sx**2) \
            * np.exp(-0.5*z**2/sz**2) \
            * np.exp(+(1j/hb)*(px*x + pz*z))
def psi0ringUnNorm(x,z,pr=p,mur=10,sg=sg):
#     return (pi**1.5 * sg * (1 + scipy.special.erf(mur/sg)))**-1 \
    return 1 \
            * np.exp(-0.5*( mur - np.sqrt(x**2 + z**2) )**2 / sg**2) \
            * np.exp(+(1j/hb) * (x**2 + z**2)**0.5 * pr)


expPGrid = np.zeros((nx,nz),dtype=complex)
for indx in range(nx):
    expPGrid[indx, :] = np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin[indx]**2 + pzlin**2))  


def psi0np(mux=10,muz=10,p0x=0,p0z=0):
    psi=np.zeros((nx,nz),dtype=complex)
    for ix in range(1,nx-1):
        x = xlin[ix]
        psi[ix][1:-1] = psi0(x,zlin[1:-1],mux,muz,p0x,p0z)
    return psi
def psi0ringNp(mur=1,sg=1,pr=p):
    psi = np.zeros((nx,nz),dtype=complex)
    for ix in range(1,nx-1):
        x = xlin[ix]
        psi[ix][1:-1] = psi0ringUnNorm(x,zlin[1:-1],pr,mur,sg)
    norm = np.sum(np.abs(psi)**2)*dx*dz
    psi *= 1/sqrt(norm)
    return psi


# + editable=true slideshow={"slide_type": ""}
def psi0ringUnNormOffset(x,z,pr=p,mur=10,sg=sg,xo=0,zo=0,pxo=0,pzo=0):
    return 1 \
            * np.exp(-0.5*( mur - np.sqrt((x-xo)**2 + (z-zo)**2) )**2 / sg**2) \
            * np.exp(+(1j/hb) * (((x-xo)**2 + (z-zo)**2)**0.5 * pr + x*pxo+z*pzo))
def psi0ringNpOffset(mur=1,sg=1,pr=p,xo=0,zo=0,pxo=0,pzo=0):
    psi = np.zeros((nx,nz),dtype=np.complex128)
    for ix in range(0,nx):
        x = xlin[ix]
        psi[ix,:] = psi0ringUnNormOffset(x,zlin,pr,mur,sg,xo,zo,pxo,pzo)
    norm = np.sum(np.abs(psi)**2)*dx*dz
    psi *= 1/sqrt(norm)
    return psi


# + editable=true slideshow={"slide_type": ""}
# psi = psi0np(5,5,0,0)
# psi = psi0np(5,5,-0.5*p,0)
# psi = psi0np(1,1,p,p)
# psi = psi0np(1,1,0,0)

# @jit(cache=True, forceobj=True)
def phiAndSWNF(psi):
    phiUN = np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))
    # superWeirdNormalisationFactorSq = np.trapz(np.trapz(np.abs(phiUN)**2, pxlin, axis=0), pzlin)
    superWeirdNormalisationFactorSq = np.sum(np.abs(phiUN)**2)*dpx*dpz
    swnf = sqrt(superWeirdNormalisationFactorSq)
    phi = phiUN/swnf
    return (swnf, phi)
    
# psi = psi0ringNp(4,2,p)
# psi = psi0ringNpOffset(5,3,p,0,5,0,p)
# psi = psi0np(2,2,0.5*p*np.cos(0),0.5*p*np.sin(0))
# psi = psi0np(mux=3,muz=3,p0x=0,p0z=0)
# psi = psi0ringNpOffset(5,3,p,0,5,0,p)
psi = psi0ringNpOffset(10,3,p,0,20,0,p)
(swnf, phi) = phiAndSWNF(psi)

t = 0

print("Super weird normalisation factor, swnf =",swnf)
print(np.sum(np.abs(phi)**2)*dpx*dpz, "|phi|**2 normalisation check")
print(np.sum(np.abs(psi)**2)*dx*dz,   "|psi|**2 normalisation check")
plot_psi(psi,False)
title="init_ring_psi"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()

plot_mom(psi,10,10,False)
title="init_ring_phi"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()

# -

v4*0.2


# + editable=true slideshow={"slide_type": ""}
# @jit(cache=True, forceobj=True)
def toMomentum(psi, swnf):
    return np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))/swnf
# @jit(cache=True, forceobj=True)
def toPosition(phi, swnf):
    return pyfftw.interfaces.numpy_fft.ifft2(np.fft.ifftshift(np.fliplr(phi*swnf)),threads=nthreads,norm='ortho')


# + editable=true slideshow={"slide_type": ""}
def plotNow(t, psi):
        print("time =", round(t*1000,4), "µs")
        print(np.sum(np.abs(psi)**2)*dx*dz,"|psi|^2")
        print(np.sum(np.abs(phi)**2)*dpx*dpz,"|phi|^2")
        plot_psi(psi)
        plot_mom(psi)

# @jit(forceobj=True, cache=True)
def numericalEvolve(
        t_init, 
        psi_init, 
        t_final, 
        tauPi  = tBraggPi, 
        tauMid = tBraggPi*5, 
        phase  = 0,
        doppd=dopd,
        print_every_t=-1, 
        final_plot=True,
        progress_bar=True, 
        V0FArg=V0F,
        kkx=kx,
        kkz=kz
    ):
    assert (print_every_t > dt or print_every_t <= 0), "print_every_t cannot be smaller than dt"
    steps = ceil((t_final - t_init) / dt) 
    t = t_init
    psi = psi_init.copy()
    (swnf, phi) = phiAndSWNF(psi)
    
#     tauMid = tauPi * 5
#     tauEnd = tauPi * 10

    def loop():
        nonlocal t
        nonlocal psi
        nonlocal phi
        # cosGrid = np.cos(2*k*xgrid + doppd*(t-tBraggCenter) + phase)
        cosGrid = np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase)
        VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * VS(t,tauMid,tauPi,V0FArg) * cosGrid )
        # VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * V0FArg * 
        #                    np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase))
        psi *= VxExpGrid
        phi = toMomentum(psi,swnf)
        phi *= expPGrid
        psi = toPosition(phi,swnf)
        psi *= VxExpGrid
        
        if print_every_t > 0 and step % round(print_every_t / dt) == 0: 
            plotNow(t,psi)
        t += dt 
        
    if progress_bar:
        for step in tqdm(range(steps)):
            loop()
    else:
        for step in range(steps):
            loop()
    
    if final_plot:
        print("ALL DONE")
        plotNow(t,psi)
    return (t,psi,phi)


# + vscode={"languageId": "raw"} active=""
# %timeit _ = numericalEvolve(0, psi0np(1,1,0,0), dt, final_plot=False, progress_bar=False)

# + active=""
# # @njit
# # @jit(cache=True, forceobj=True)
# def numericalEvolveNumba(
#         t_init = 0, 
#         psi_init = np.array([]), 
#         t_final =0, 
#         tauPi  = tBraggPi, 
#         tauMid = tBraggPi*5, 
#         phase  = 0,
#         doppd=dopd,
#         print_every_t=-1, 
#         final_plot=True,
#         progress_bar=True, 
#         V0FArg=V0F,
#         kkx=kx,
#         kkz=kz
#     ):
#     assert (print_every_t > dt or print_every_t <= 0), "print_every_t cannot be smaller than dt"
#     steps = ceil((t_final - t_init) / dt) 
#     t = t_init
#     psi = psi_init.copy()
#     (swnf, phi) = phiAndSWNF(psi)
#     
#     for step in range(steps):
#         # cosGrid = np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase)
#         VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * VS(t,tauMid,tauPi,V0FArg) * 
#                            np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase))
#         # VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * V0FArg * 
#         #                    np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase))
#         psi *= VxExpGrid
#         phi = toMomentum(psi,swnf)
#         phi *= expPGrid
#         psi = toPosition(phi,swnf)
#         psi *= VxExpGrid
#     
#     return (t,psi,phi)

# + active=""
# %timeit _ = numericalEvolveNumba(0, psi0np(1,1,0,0), dt, final_plot=False, progress_bar=False)
# -

def freeEvolve(
    t_init,
    psi,
    t_final,
    final_plot=True,
    logging=False,
    ):
    Dt = t_final-t_init
    
    (swnf, phi) = phiAndSWNF(psi)
    
    if logging: print("checking this value is small ", -(1j/hb) * (0.5/m3) * (Dt)*pxmax)
    expPGridLong = np.zeros((nx,nz),dtype=complex)
    for indx in range(nx):
        expPGridLong[indx, :] = np.exp(-(1j/hb) * (0.5/m3) * (Dt) * (pxlin[indx]**2 + pzlin**2)) 
    
    phi *= expPGridLong
    psi = toPosition(phi,swnf)
    
    if final_plot:
        plotNow(t_final,psi)
        
    return (t_final, psi, phi)


_ = freeEvolve(0,psi0np(1,1,0,0),0.1,final_plot=False,logging=True)



# ## Pulse Scan Test Run (MUST RUN)

# short test run
_ = numericalEvolve(0, psi0np(3,3,0.5*p,0), 2*dt,progress_bar=False,final_plot=False)


# + editable=true slideshow={"slide_type": ""}
def scanTauPiInnerEval(tPi, 
                       logging=True, progress_bar=True, 
                       ang=0, pmom=p, doppd=dopd, V0FArg=V0F, kkx=kx, kkz=kz,
                       halo_xr=10
                      ):
    tauPi  = tPi
    tauMid = tauPi / 2 
    tauEnd = tauPi 
    if logging:
        print("Testing parameters")
        print("tauPi =", round(tPi,6), "    \t tauMid =", round(tauMid,6), " \t tauEnd = ", round(tauEnd,6))
    # output = numericalEvolve(0, psi0np(2,2,pmom*np.cos(ang),pmom*np.sin(ang)), 
    output = numericalEvolve(0, 
                             # psi0np(mux=3,muz=3,p0x=0,p0z=0),
                             # psi0ringNpOffset(5,3,pmom,0,5,0,pmom), 
                             psi0ringNpOffset(halo_xr,3,pmom,0,halo_xr,0,pmom), 
                             tauEnd, tauPi, tauMid, doppd=doppd, 
                             final_plot=logging,progress_bar=progress_bar,
                             V0FArg=V0FArg,kkx=kkx,kkz=kkz
                            )
#     if pbar != None: pbar.update(1)
    gc.collect()
    return output


# -

# This is to get an estimate of long it takes to simulate 10us
# This takes about 1 minute
tPiScanTime10usStart = datetime.now()
_ = scanTauPiInnerEval(0.010, False, True,0,p,1*dopd,VR)
tPiScanTime10usEnd = datetime.now()
tPiScanTime10usDelta = tPiScanTime10usEnd - tPiScanTime10usStart
l.info(f"""Time to simulate 1us: {tPiScanTime10usDelta}""")

# +
tPDelta = 2*dt  # positive +, note I want tPiTest in decending order 
# tPiTest = np.append(np.arange(0.5,0.1,-tPDelta), 0) # note this is decending
tPiTest = np.arange(0.04,0.0006-0*tPDelta,-tPDelta)
    # tPiTest = np.arange(dt,3*dt,dt)
l.info(f"len(tPiTest) = {len(tPiTest)}, max={tPiTest[0]*1000}, min={tPiTest[-1]*1000} us")
l.info(f"tPiTest: {round(tPiTest[0],6)}, {round(tPiTest[1],6)}, ..., {round(tPiTest[-2],6)}, {round(tPiTest[-1],6)}")
l.info(f"""psi size is {round(sys.getsizeof(psi)/1024**2,3)} MB
need {round(sys.getsizeof(psi)/1024**3 * len(tPiTest),3)} GB RAM to tPiOutput""")

plt.figure(figsize=(12,5))
def plot_inner_helper():
    tPiScanTotalSimMS = 0
    for (i, tauPi) in enumerate(tPiTest):
        if tauPi == 0: continue
        tauMid = tauPi / 2 
        tauEnd = tauPi * 1
        tPiScanTotalSimMS += tauEnd
        tlinspace = np.arange(0,tauEnd,dt)
        # plt.plot(tlinspace, VBF(tlinspace, tauMid, tauPi),
        #          linewidth=0.5,alpha=0.9
        #     )
        plt.plot(tlinspace, VS(tlinspace, tauMid, tauPi),
                 linewidth=0.5,alpha=0.9
            )
    return tPiScanTotalSimMS

plt.subplot(2,1,1)
tPiScanTotalSimMS = plot_inner_helper()
l.info(f"roughtly can finish in {round((100*tPiScanTime10usDelta*tPiScanTotalSimMS).total_seconds()/(0.7*N_JOB2)/60, 3)} min")
plt.ylabel("$V(t)$")

plt.subplot(2,1,2)
plot_inner_helper()
plt.xlim([0,0.002])
plt.xlabel("$t \ (ms)$ ")
plt.ylabel("$V(t)$")


title="bragg_strength_V0"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

# -

tPiTest



assert False, "catch run all"

# ## Manual Scan

# + editable=true slideshow={"slide_type": ""}
tPiScanTimeStart = datetime.now()
tPiOutput = Parallel(n_jobs=N_JOB2)(
    delayed(lambda i: (i, scanTauPiInnerEval(i, False, False,0,p,0.5*dopd,VR)[:2]) )(i) 
    for i in tqdm(tPiTest)
) 
tPiScanTimeEnd = datetime.now()
tPiScanTimeDelta = tPiScanTimeEnd-tPiScanTimeStart
l.info(f"""Time to run one scan: {tPiScanTimeDelta}""")
# -

os.makedirs(output_prefix+"tPiScan", exist_ok=True)
with pgzip.open(output_prefix+"tPiScan/"+f"tPiOutput"+output_ext,'wb', thread=4, blocksize=2*10**8) as file:
    pickle.dump(tPiOutput, file) 
gc.collect()

# + editable=true slideshow={"slide_type": ""}
# psi = tPiOutput[-30][1][1]
ind = 80
psi = tPiOutput[ind][1][1]
print(round(tPiOutput[ind][0]*1000, 3))
# psi = tPiTestRun[1]L
# psi = testFreeEv1[1]
# plot_psi(psi)
(swnf, phi) = phiAndSWNF(psi)
plot_mom(psi,8,8)

# + [markdown] editable=true slideshow={"slide_type": ""}
# #### zcut method calculations

# + editable=true slideshow={"slide_type": ""}
# hbar_k_transfers = np.arange(-4,4+1,+2)
hbar_k_transfers = np.arange(-10-1,10+2,+2)
# pzlinIndexSet = np.zeros((len(hbar_k_transfers), len(pxlin)), dtype=bool)
pxlinIndexSet = np.zeros((len(hbar_k_transfers), len(pzlin)), dtype=bool)
cut_p_width = 5*dpz/p
for (j, hbar_k) in enumerate(hbar_k_transfers):
    # pzlinIndexSet[j] = abs(pxlin/(hb*k) - hbar_k) <= cut_p_width
    pxlinIndexSet[j] = abs(pzlin/p + hbar_k) <= cut_p_width
    # print(i,hbar_k)
l.info(f"""hbar_k_transfers = {hbar_k_transfers}
np.sum(pxlinIndexSet,axis=1) = {np.sum(pxlinIndexSet,axis=1)}""")

# + editable=true slideshow={"slide_type": ""}
plt.figure(figsize=(4,4))
plt.imshow(pxlinIndexSet.T,interpolation='none',aspect=0.1,extent=[-2,2,-pzmax/p,pzmax/p])
# plt.imshow(pzlinIndexSet,interpolation='none',aspect=5)
# plt.axvline(x=1001, linewidth=1, alpha=0.7)

title="hbar_k_pxlin_integration_range"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()

# + editable=true slideshow={"slide_type": ""}
# phiDensityGrid = np.zeros((len(tPiTest), pxlin.size))
phiDensityGrid = np.zeros((len(tPiTest), pzlin.size))
phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))

for i in tqdm(range(len(tPiTest))):
    item = tPiOutput[i]
    (swnf, phi) = phiAndSWNF(item[1][1])
    phiAbsSq = np.abs(phi)**2
    # phiX = np.trapz(phiAbsSq, pzlin,axis=1)
    phiZ = np.trapz(phiAbsSq, pzlin,axis=0)
    # phiDensityGrid[i] = phiX
    phiDensityGrid[i] = phiZ

    for (j, hbar_k) in enumerate(hbar_k_transfers):
        # index = pzlinIndexSet[j]
        index = pxlinIndexSet[j]
        # phiDensityGrid_hbark[i,j] = np.trapz(phiX[index], pxlin[index])
        # phiDensityGrid_hbark[i,j] = np.trapz(phiZ[index], pzlin[index])
        phiDensityGrid_hbark[i,j] = np.sum(phiZ[index])

# + editable=true slideshow={"slide_type": ""}
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

nxm = int((nx-1)/2)
nx2 = int((nx-1)/2)
mom_dist_at_diff_angle_phi_asss=(2*pxmax/p)/len(tPiOutput)
mom_dist_at_diff_angle_den_asss=(hbar_k_transfers[-1]-hbar_k_transfers[0])/len(tPiOutput)
plt.imshow(#phiDensityGrid[:,nxm-nx2:nxm+nx2], 
            phiDensityGrid, 
           # extent=[pxlin[nxm-nx2]/(hb*k),pxlin[nxm+nx2]/(hb*k),0,len(tPiTest)], 
           extent=[pzlin[nxm-nx2]/(hb*k),pzlin[nxm+nx2]/(hb*k),0,len(tPiTest)], 
           interpolation='none',aspect=mom_dist_at_diff_angle_phi_asss)
# plt.imshow(phiDensityGrid, 
#            extent=[-pxmax/(hb*k),pxmax/(hb*k),1,len(tPiTest)+1], 
#            interpolation='none',aspect=1)
# ax = plt.gca()
# for t in tPiTest:
#     plt.axhline(y=t/dt,color='white',alpha=1,linewidth=0.05,linestyle='-')
ax = plt.gca()
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.ylabel("$dt =$"+str(dt*1000) + "$\mu \mathrm{s}$")
# plt.xlabel("$p_x \ (\hbar k)$")
plt.xlabel("$p_z \ (\hbar k)$")


plt.subplot(1,2,2)
plt.imshow(phiDensityGrid_hbark, 
           extent=[hbar_k_transfers[0],hbar_k_transfers[-1],0,len(tPiTest)], 
           interpolation='none',aspect=mom_dist_at_diff_angle_den_asss)
# plt.xlabel("$p_x \ (\hbar k)$ integrated block")
plt.xlabel("$p_z \ (\hbar k)$ integrated block")

title="mom_dist_at_diff_angle"
plt.savefig("output/"+title+".pdf", dpi=600)
plt.savefig("output/"+title+".png", dpi=600)
plt.show()

# + editable=true slideshow={"slide_type": ""}
# phiDensityNormFactor = np.sum(phiDensityGrid_hbark,axis=1)
phiDensityNormFactor = np.trapz(phiDensityGrid_hbark,axis=1)
# phiDensityNormed = np.zeros(phiDensityGrid_hbark.shape)
# for i in range(len(hbar_k_transfers)):
#     phiDensityNormed[:,i] = phiDensityGrid_hbark[:,i]/phiDensityNormFactor[i]
phiDensityNormed = phiDensityGrid_hbark / phiDensityNormFactor[:, np.newaxis]
# -

gc.collect()

# +
# phiDensityNormFactor

# + editable=true slideshow={"slide_type": ""}
os.makedirs(output_prefix+"tPiScan", exist_ok=True)

# + editable=true slideshow={"slide_type": ""}
plt.figure(figsize=(11,4))
for (i, hbar_k) in enumerate(hbar_k_transfers):
    if abs(hbar_k) >5: continue
    if   hbar_k > 0: style = '+-'
    elif hbar_k < 0: style = 'x-'
    else:            style = '.-'
    plt.plot(tPiTest*1000, phiDensityNormed[:,i],
             style, linewidth=1,alpha=0.4, markersize=5,
             label=str(hbar_k)+"$\hbar k$",
            )

plt.legend(loc=2,ncols=2)
plt.ylabel("Population Fraction (Normalised by \n $\int |\phi(p)|^2 dp$ integrating cuts ($\pm$"+str(round(cut_p_width,3))+"$\hbar k$))")
plt.xlabel("Pulse width $(\mu s)$")
# plt.axhline(y=np.cos(pi  )**2,color='gray',linewidth=1,alpha=0.5)  # 2*pi pulse
# plt.axhline(y=np.cos(pi/2)**2,color='c',linewidth=1,alpha=0.5)     # pi   pulse
# plt.axhline(y=np.cos(pi/4)**2,color='violet',linewidth=1,alpha=0.5) # pi/2 pulse
# plt.axhline(y=np.cos(pi/8)**2,color='orange',linewidth=1,alpha=0.5)# pi/4 pulse

# plt.axvline(x=tPiTest[81]*1000,color='c',linewidth=1,alpha=0.5)      # pi   pulse
# plt.axvline(x=tPiTest[90]*1000,color='violet',linewidth=1,alpha=0.5) # pi/2 pulse
# plt.axvline(x= 9*dt*1000,color='orange',linewidth=1,alpha=0.5)  # pi/4 pulse

# plt.text((1+39)*dt*1000, 1, "$\pi$",color='c')
# plt.text((1+21)*dt*1000, 1, "$\pi/2$",color='violet')
# plt.text((1+ 9)*dt*1000, 1, "$\pi/4$",color='orange')

plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1/4))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.1))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1/4))
plt.xlim([0,30.5])

title = "bragg_pulse_duration_test_labeled"
plt.savefig(output_prefix+"tPiScan/"+title+".pdf", dpi=600,bbox_inches='tight')
plt.savefig(output_prefix+"tPiScan/"+title+".png", dpi=600,bbox_inches='tight')

plt.show()

# +
hbarkInI = next(iter(np.where(hbar_k_transfers == +1)[0]), None)  # Index of original state 
hbarkInd = next(iter(np.where(hbar_k_transfers == -1)[0]), None)  # Index of target state
hbarkInA = next(iter(np.where(hbar_k_transfers == -2)[0]), None)  # target -2hbk for initial scattering
hbarkInB = next(iter(np.where(hbar_k_transfers == +2)[0]), None)  # target +2hbk

l.info(f"""target state: {hbar_k_transfers[hbarkInd] if hbarkInd is not None else 'NA'}, original state: {hbar_k_transfers[hbarkInI] if hbarkInI is not None else 'NA'}
target A: {hbar_k_transfers[hbarkInA] if hbarkInA is not None else 'NA'}, target B: {hbar_k_transfers[hbarkInB] if hbarkInB is not None else 'NA'} """)
# currently (May 2024), by design, only one set will get used at a time. 
# -

# #### Angle Transfer Stuff

# +
# momAngMask = np.zeros((nx,nz))
# for (xi, px) in enumerate(pxlin):
#     momAngMask[xi,:] = np.exp(  -((px - p*sin(mA))**2 + (-pzlin - p - p*cos(mA))**2) / (2*5*dpz)**2 )
# -

def momAngMaskCombo(mA,pwid=3*dpz):
    #REVIEW: this thing might contain a +/- p definition error, or could mean somewhere in the code is problematic
    momAngMaskUR = np.exp(-((pxlin[:,np.newaxis] - p*sin(mA))**2 + (-pzlin[np.newaxis,:] - p - p*cos(mA))**2) / (pwid)**2)
    momAngMaskUL = np.exp(-((pxlin[:,np.newaxis] + p*sin(mA))**2 + (-pzlin[np.newaxis,:] - p + p*cos(mA))**2) / (pwid)**2)
    momAngMaskDR = np.exp(-((pxlin[:,np.newaxis] - p*sin(mA))**2 + (-pzlin[np.newaxis,:] + p - p*cos(mA))**2) / (pwid)**2)
    momAngMaskDL = np.exp(-((pxlin[:,np.newaxis] + p*sin(mA))**2 + (-pzlin[np.newaxis,:] + p + p*cos(mA))**2) / (pwid)**2)
    return (momAngMaskUR, momAngMaskUL, momAngMaskDR, momAngMaskDL)


def ct_cmap(base_cmap):
    cmap = matplotlib.colormaps[base_cmap]
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = np.linspace(0, 1, cmap.N)  # Set transparency gradient
    return ListedColormap(colors)


def momAngFVal(mA,pwid,phi):
    (maUR, maUL, maDR, maDL) = momAngMaskCombo(mA, pwid)
    return (
        np.sum(np.abs(phi)**2 * maUR * dpz * dpx),
        np.sum(np.abs(phi)**2 * maUL * dpz * dpx),
        np.sum(np.abs(phi)**2 * maDR * dpz * dpx),
        np.sum(np.abs(phi)**2 * maDL * dpz * dpx)
    )


# +
momAngList = np.arange(0,180,1)*pi/180
# mA = momAngList[45]
# momAngMask = np.exp(-((pxlin[:,np.newaxis] - p*np.sin(mA))**2 + (-pzlin[np.newaxis, :] - p - p*np.cos(mA))**2) / (2 * 5 * dpz)**2)
(maUR, maUL, maDR, maDL) = momAngMaskCombo(momAngList[45], pwid=5*dpz)
maTemp = momAngFVal(momAngList[45],pwid=5*dpz, phi=phi)
l.info(maTemp)

# momAngResults = np.zeros((momAngList.shape[0], 4))
# for (mAi, mAv) in tqdm(enumerate(momAngList), total=momAngList.shape[0]):
#     momAngResults[mAi] = np.array(momAngFVal(mAv, pwid=5*dpz, phi=phi))
momAngResults = Parallel(n_jobs=-2)( 
        delayed(lambda mAv: np.array(momAngFVal(mAv, pwid=5*dpz, phi=phi)))(mAv)
        for mAv in tqdm(momAngList)
    )
momAngResults = np.array(momAngResults)

# +
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.imshow(np.abs(phi.T)**2, cmap='Greys', alpha=0.6, extent=np.array([-pxmax,+pxmax,-pzmax,+pzmax])/p, interpolation='None')
plt.imshow(maUR.T, cmap=ct_cmap('Greens'), alpha=0.9, extent=np.array([-pxmax,+pxmax,-pzmax,+pzmax])/p, interpolation='None')
plt.imshow(maUL.T, cmap=ct_cmap('Oranges'), alpha=0.9, extent=np.array([-pxmax,+pxmax,-pzmax,+pzmax])/p, interpolation='None')
plt.imshow(maDR.T, cmap=ct_cmap('Blues'), alpha=0.9, extent=np.array([-pxmax,+pxmax,-pzmax,+pzmax])/p, interpolation='None')
plt.imshow(maDL.T, cmap=ct_cmap('Purples'), alpha=0.9, extent=np.array([-pxmax,+pxmax,-pzmax,+pzmax])/p, interpolation='None',label="DL")

plt.xlim([-3, +3])
plt.ylim([-3, +3])
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1/4))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1/4))
plt.xlabel("$p_x (\hbar k)$")
plt.ylabel("$p_z (\hbar k)$")
plt.text(+0.80,+1.80,"|UR⟩")
plt.text(+0.80,-0.20,"|DR⟩")
plt.text(-1.45,+0.10,"|UL⟩")
plt.text(-1.45,-1.85,"|DL⟩")


plt.subplot(1,2,2)
plt.imshow(momAngResults.T, extent=[momAngList[0]*180/pi, momAngList[-1]*180/pi,-0.5,3.5],interpolation='None',aspect='auto', rasterized=True)
# plt.gca().set_rasterized(True)
plt.yticks(range(4), labels=["DL","DR","UL","UR"])
# plt.xticks([180/6 for i in range(7)], labels=[f"{round(i*180/6)}" for i in range(7)])
# plt.grid(axis='x',alpha=0.5,linewidth=0.5)
# below line sets the minor ticks =
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=30))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
plt.xlabel("Polar angle (deg) from halo north pole")
plt.ylabel("Population overlap $|⟨XY|\psi⟩|^2$ \n($|\psi_\mathrm{total}|^2=1$ normalisation)")
plt.colorbar()
# plt.xlim([0,pi])


title = "halo_mom_ang_labels"
plt.savefig(output_prefix+"tPiScan/"+title+".pdf", dpi=600,bbox_inches='tight')
plt.savefig(output_prefix+"tPiScan/"+title+".png", dpi=600,bbox_inches='tight')
plt.show()
# -

momAngPiScan = np.zeros((len(tPiTest), len(momAngList), 4))
for (ti, tt) in tqdm(enumerate(tPiTest), total=len(tPiTest)):
    item = tPiOutput[ti]
    (swnf, phi) = phiAndSWNF(item[1][1])
    momAngResults = Parallel(n_jobs=-4)( 
        delayed(lambda mAv: np.array(momAngFVal(mAv, pwid=5*dpz, phi=phi)))(mAv)
        for mAv in momAngList
    )
    momAngResults = np.array(momAngResults)
    momAngPiScan[ti] = momAngResults
del item, phi, swnf, momAngResults
gc.collect()

momAngPiScan.shape

# +
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.plot(momAngList*180/pi, momAngPiScan[80,:,0], label=f"UR {round(tPiTest[80]*1000,1)}$\mu s$", color='b', linestyle='-', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[80,:,3], label=f"DR {round(tPiTest[80]*1000,1)}$\mu s$", color='r', linestyle='-', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[85,:,0], label=f"UR {round(tPiTest[85]*1000,1)}$\mu s$", color='b', linestyle='--', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[85,:,3], label=f"DR {round(tPiTest[85]*1000,1)}$\mu s$", color='r', linestyle='--', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[90,:,0], label=f"UR {round(tPiTest[85]*1000,1)}$\mu s$", color='b', linestyle='-.', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[90,:,3], label=f"DR {round(tPiTest[85]*1000,1)}$\mu s$", color='r', linestyle='-.', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[95,:,0], label=f"UR {round(tPiTest[85]*1000,1)}$\mu s$", color='b', linestyle=':', alpha=0.5)
plt.plot(momAngList*180/pi, momAngPiScan[95,:,3], label=f"DR {round(tPiTest[85]*1000,1)}$\mu s$", color='r', linestyle=':', alpha=0.5)
# plt.plot(momAngList*180/pi, momAngPiScan[91,:,0], label=f"UR {round(tPiTest[91]*1000,1)}$\mu s, \pi/2$", color='b', linestyle='-.', alpha=0.5)
# plt.plot(momAngList*180/pi, momAngPiScan[91,:,3], label=f"DR {round(tPiTest[91]*1000,1)}$\mu s, \pi/2$", color='r', linestyle='-.', alpha=0.5)
# plt.plot(momAngList*180/pi, momAngPiScan[97,:,0], label=f"UR {round(tPiTest[97]*1000,1)}$\mu s, \pi/4$", color='b', linestyle=':', alpha=0.5)
# plt.plot(momAngList*180/pi, momAngPiScan[97,:,3], label=f"DR {round(tPiTest[97]*1000,1)}$\mu s, \pi/4$", color='r', linestyle=':', alpha=0.5)
plt.xlabel("Polar angle (deg) from halo north pole")
plt.ylabel("Population overlap")
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.0005))
plt.legend(loc='upper center',fontsize='x-small', ncol=4,bbox_to_anchor=(0.5,1.13))
# plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(tPiTest*1000,momAngPiScan[:,90,0], label="UR at 90$^\circ$", color='b', linestyle='-', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,90,2], label="DR at 90$^\circ$", color='r', linestyle='-', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,75,0], label="UR at 75$^\circ$", color='b', linestyle='--', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,75,2], label="DR at 75$^\circ$", color='r', linestyle='--', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,60,0], label="UR at 60$^\circ$", color='b', linestyle='-.', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,60,2], label="DR at 60$^\circ$", color='r', linestyle='-.', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,45,0], label="UR at 45$^\circ$", color='b', linestyle=':', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,45,2], label="DR at 45$^\circ$", color='r', linestyle=':', alpha=0.5)
plt.xlabel("Pulse width ($\mu s$)")
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.0005))
# plt.ylabel("Population ($|\psi_\mathrm{total}|^2=1$ normalisation)")
plt.ylabel("Population overlap")
plt.legend(loc='upper center',fontsize='x-small', ncol=4,bbox_to_anchor=(0.5,1.115))


title = "halo_mom_ang_scan"
plt.savefig(output_prefix+"tPiScan/"+title+".pdf", dpi=600,bbox_inches='tight')
plt.savefig(output_prefix+"tPiScan/"+title+".png", dpi=600,bbox_inches='tight')
plt.show()

# +
ang = 45
plt.plot(tPiTest*1000,momAngPiScan[:,ang,0], label=f"UR at {ang}", color='b', linestyle='--', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,ang,2], label=f"DR at {ang}", color='r', linestyle='--', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,ang,0]+momAngPiScan[:,ang,2], label=f"UR+DR at {ang}", color='g', linestyle='-', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,ang,1], label=f"UL at {ang}", color='b', linestyle=':', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,ang,3], label=f"DL at {ang}", color='r', linestyle=':', alpha=0.5)
plt.plot(tPiTest*1000,momAngPiScan[:,ang,1]+momAngPiScan[:,ang,3], label=f"UL+DL at {ang}", color='y', linestyle='-', alpha=0.5)

plt.xlabel("Pulse width ($\mu s$)")
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.0005))
# plt.ylabel("Population ($|\psi_\mathrm{total}|^2=1$ normalisation)")
plt.ylabel("Population overlap")
plt.legend(loc='upper center',fontsize='x-small', ncol=4,bbox_to_anchor=(0.5,1.115))
plt.show()

# +
plt.figure(figsize=(11,3))
gam = 0.07
omP = pi/6
ts = np.linspace(0,30,1000)
plt.plot(ts, 0+np.sin(omP*ts)**2, label="|U⟩ \t $  \sin^2(\omega t)                $", alpha=0.3,color='b')
plt.plot(ts, 1-np.sin(omP*ts)**2, label="|D⟩ \t $1-\sin^2(\omega t)=cos^2(\omega t)$", alpha=0.3,color='r')
plt.plot(ts, 0+np.sin(omP*ts)**2*np.exp(-gam*ts), label="|U⟩ \t $  \sin^2(\omega t)e^{-\gamma t }$", alpha=0.9, linestyle='--',color='b')
plt.plot(ts, 1-np.sin(omP*ts)**2*np.exp(-gam*ts), label="|D⟩ \t $1-\sin^2(\omega t)e^{-\gamma t }$", alpha=0.9, linestyle='--',color='r')

plt.legend(loc=7)
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1/4))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.1))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1/2))
plt.xlim([0,30])
plt.xlabel("Pulse width ($\mu s$)")
plt.ylabel("Population Fraction")

title = "halo_mom_trans_model"
plt.savefig(output_prefix+"tPiScan/"+title+".pdf", dpi=600,bbox_inches='tight')
plt.savefig(output_prefix+"tPiScan/"+title+".png", dpi=600,bbox_inches='tight')

plt.show()
# -



# #### Dump / Load

with pgzip.open(output_prefix+"tPiScan/"+f"tPiTest"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(tPiTest, file) 
with pgzip.open(output_prefix+"tPiScan/"+f"momAngList"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(momAngList, file) 
with pgzip.open(output_prefix+"tPiScan/"+f"momAngPiScan"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(momAngPiScan, file) 

# del momAngPiScan
# with pgzip.open('/Volumes/tonyNVME Gold/oneParticleSim/20240704-165506-TFF/tPiScan 2*dopd/momAngPiScan.pgz.pkl' , 'rb', thread=8) as file:
#     momAngPiScan = pickle.load(file)
with pgzip.open('/Volumes/tonyNVME Gold/oneParticleSim/20240703-225141-TFF/tPiScan dopd*0/momAngPiScan.pgz.pkl' , 'rb', thread=8) as file:
    momAngPiScan = pickle.load(file)
with pgzip.open('/Volumes/tonyNVME Gold/oneParticleSim/20240703-225141-TFF/tPiScan dopd*0/tPiTest.pgz.pkl' , 'rb', thread=8) as file:
    tPiTest = pickle.load(file)
# del tPiOutput
# with pgzip.open('/Volumes/tonyNVME Gold/oneParticleSim/20240703-225141-TFF/tPiScan/tPiOutput1VR.pgz.pkl' , 'rb', thread=8) as file:
#     tPiOutput = pickle.load(file)
# del tPiOutput

sys.getsizeof(tPiOutput)/1024**2

# del tPiOutput, momAngPiScan
gc.collect()

# ### Pulse eff peaks search

thetaList = np.arange(0,16+1,1)*pi/4
popTarList = np.sin(thetaList/2)**2
popIniList = np.cos(thetaList/2)**2
l.info(f"""thetaList = {thetaList}
popTarList = {popTarList}
popIniList = {popIniList}""")


def find_pulses(phiDN, thetaLis, makeFig=False):
    peaks_ind = {}
    peaks_time = {}
    peaks_helper = {}
    for (ind, theta) in enumerate(thetaLis):
        popTar = np.sin(theta/2)**2
        popIni = np.cos(theta/2)**2
        # print(popTar, popIni)
        searchHelper = np.abs(phiDN[:,hbarkInd]-popTar)+np.abs(phiDN[:,hbarkInI]-popIni)
        found_results = np.flip(scipy.signal.find_peaks(-searchHelper)[0])
        peaks_ind[ind] = found_results
        peaks_time[ind] = tPiTest[found_results]
        peaks_helper[ind] = searchHelper[found_results] 
        if makeFig:
            l.info(f"""ind = {ind}, theta = {theta}, popTar = {popTar}, popIni = {popIni} 
            found_results: {found_results}
                tPiTest[...]:      {tPiTest[found_results]}
                searchHelper[...]: {searchHelper[found_results]}""")
            plt.plot(tPiTest*1000,searchHelper,'.-',alpha=0.7,label='helper')
            plt.plot(tPiTest*1000,np.abs(phiDN[:,hbarkInd]-popTar),'x-',alpha=0.3,label="-1")
            plt.plot(tPiTest*1000,np.abs(phiDN[:,hbarkInI]-popIni),'+-',alpha=0.3,label="+1")
            plt.legend(ncols=3)
            plt.ylabel("fraction")
            plt.xlabel("$t_\pi \ (\mu s)$")
            plt.show()
    return (peaks_ind, peaks_time, peaks_helper)


find_pulses(phiDensityNormed, thetaList, makeFig=True)

l.info(find_pulses(phiDensityNormed, thetaList))









# #### Some old code

# + active=""
# indPDNPi = np.argmax(phiDensityNormed[:,hbarkInd])
# l.info(f"""max transfer (π) to -1hbk at σt {tPiTest[indPDNPi]*1000} μs 
# with efficiency to -1hbk: {phiDensityNormed[indPDNPi,hbarkInd]}""")

# + active=""
# SC_searchHelper=np.abs(phiDensityNormed[:,hbarkInA]-0.5)+np.abs(phiDensityNormed[:,hbarkInB]-0.5)
# indPDNSC = np.argmin(SC_searchHelper)
# l.info(f"""max scattering to ±2hbk at σt {round(tPiTest[indPDNSC]*1000,6)} μs 
# with transfer fraction to -2hbk of {phiDensityNormed[indPDNSC,hbarkInA]}
# with transfer fraction to +2hbk of {phiDensityNormed[indPDNSC,hbarkInB]}""")

# + active=""
# pi2searchHelper=np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5)
# indPDNPM = np.argmin(pi2searchHelper)
# l.info(f"""max mirror (π/2) between ±1hbk at σt {tPiTest[indPDNPM]*1000} μs 
# with transfer fraction to -1hbk of {phiDensityNormed[indPDNPM,hbarkInd]}
# with transfer fraction to +1hbk of {phiDensityNormed[indPDNPM,hbarkInI]}""")

# + active=""
# pi2searchHelper=np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5)
# indPDNPM = np.argmin(pi2searchHelper)
# l.info(f"""max mirror (π/2) between ±1hbk at σt {tPiTest[indPDNPM]*1000} μs 
# with transfer fraction to -1hbk of {phiDensityNormed[indPDNPM,hbarkInd]}
# with transfer fraction to +1hbk of {phiDensityNormed[indPDNPM,hbarkInI]}""")
# -



# + active=""
# plt.plot(tPiTest*1000,SC_searchHelper,'.-',alpha=0.7,label='helper')
# plt.plot(tPiTest*1000,np.abs(phiDensityNormed[:,hbarkInA]-0.5),'x-',alpha=0.3,label="-1")
# plt.plot(tPiTest*1000,np.abs(phiDensityNormed[:,hbarkInB]-0.5),'+-',alpha=0.3,label="+1")
# plt.legend(ncols=3)
# plt.ylabel("fraction")
# plt.xlabel("$t_\pi \ (\mu s)$")
# plt.show()

# + active=""
# l.info(f"""indPDNPi = {indPDNPi} \ttPiTest[indPDNPi] = {round(tPiTest[indPDNPi]*1000,2)} μs \teff {round(phiDensityNormed[indPDNPi,hbarkInd],4)}
# indPDNPM = {indPDNPM} \ttPiTest[indPDNPM] = {round(tPiTest[indPDNPM]*1000,2)} μs \tef- {round(phiDensityNormed[indPDNPM,hbarkInd],4)} \tef+ {round(phiDensityNormed[indPDNPM,hbarkInI],4)}""")

# + active=""
# (indPDNPi, tPiTest[indPDNPi], phiDensityNormed[indPDNPi,hbarkInd])

# + active=""
# (indPDNPM, tPiTest[indPDNPM], phiDensityNormed[indPDNPM,hbarkInd], phiDensityNormed[indPDNPM,hbarkInI])
# -



# + active=""
# np.flip(scipy.signal.find_peaks(-SC_searchHelper)[0])

# + active=""
# tPiTest[np.flip(scipy.signal.find_peaks(-SC_searchHelper)[0])]

# + active=""
# SC_searchHelper[np.flip(scipy.signal.find_peaks(-SC_searchHelper)[0])]
# -





# #### Export frames run test

def tPiTestFrameExportHelper(ti, tPi, output_prefix_tPiVscan, skipMod=1):
    return (None, None)


# + editable=true slideshow={"slide_type": ""} active=""
# tPiScanOutputTimeStart = datetime.now()
# os.makedirs(output_prefix+"tPiScan", exist_ok=True)
# tPiTFEHSM = 100000000
# N_JOBS_PLT = -3
# l.info(f"tPiTFEHSM = {tPiTFEHSM}, f={len(tPiTest)//tPiTFEHSM}, N_JOBS_PLT = {N_JOBS_PLT}")
# def tPiTestFrameExportHelper(ti, tPi, output_prefix_tPiVscan, skipMod=1):
#     if ti % skipMod != 0: return (None, None)
#     psi = tPiOutput[ti][1][1]
#     plot_psi(psi, False)
#     plt.savefig(output_prefix_tPiVscan+f"/psi-({ti},{tPi}).png",dpi=600)
#     # tPiOutputFramesDir.append(output_prefix+f"tPiScan/psi-({ti},{tPi}).png")
#     plt.close()
#     plot_mom(psi,5,5,False)
#     plt.savefig(output_prefix_tPiVscan+f"/phi-({ti},{tPi}).png",dpi=600)
#     plt.close()
#     plt.cla() 
#     plt.clf() 
#     plt.close('all')
#     # plt.ioff() # idk, one of these should clear memroy issues?
#     time.sleep(0.01)
#     # gc.collect()
#     return(output_prefix_tPiVscan+f"/psi-({ti},{tPi}).png", 
#            output_prefix_tPiVscan+f"/phi-({ti},{tPi}).png")
#
# tPiOutputFramesDir = Parallel(n_jobs=N_JOBS_PLT)(
#     delayed(tPiTestFrameExportHelper)(ti, tPiTest[ti], output_prefix+"tPiScan", skipMod=tPiTFEHSM)
#     for ti in tqdm(range(len(tPiTest)))
# )
# tPiScanOutputTimeEnd = datetime.now()
# tPiScanOutputTimeDelta = tPiScanOutputTimeEnd-tPiScanOutputTimeStart
# l.info(f"""Time to output one scan: {tPiScanOutputTimeDelta}""")
# gc.collect()

# + active=""
# tPiOutputFramesDir

# + editable=true slideshow={"slide_type": ""}
tPiScanOutputTimeDelta = timedelta(0)

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## Intensity Scan

# + editable=true slideshow={"slide_type": ""}
isDelta = 0.1
intensityScan = np.arange(0.05,1+isDelta,isDelta)
l.info(f"""len(intensityScan): {len(intensityScan)}
intensityScan: {intensityScan}""")
omegaRabiScan = (linewidth*np.sqrt(intensityScan/intenSat/2))**2 /2/detuning
l.info(f"omegaRabiScan: {omegaRabiScan}")
VRScan = 2*hb*omegaRabiScan*0.001
l.info(f"VRScan: {VRScan}")
l.info(f"VRScan/VR: {VRScan/VR}")
# -

l.info(f"""len(intensityScan) = {len(intensityScan)}
Each scan takes time roughtly {tPiScanTimeDelta.seconds}s + {tPiScanOutputTimeDelta.seconds}s  
Estimate total scan time: {(tPiScanTimeDelta+tPiScanOutputTimeDelta)*len(intensityScan)}""")

intensityScanParamNotes = []
for i in range(len(intensityScan)):
    intensityScanParamNotes.append((i, intensityScan[i], omegaRabiScan[i], VRScan[i]))

l.info(f"""intensityScanParamNotes
(i, intensityScan[i], omegaRabiScan[i], VRScan[i]): 
{intensityScanParamNotes}""".replace("), (", "),\n("))

intensityWidthGrid = []
for (VRi,VRs) in enumerate(VRScan):
    tempTRow = []
    for (ti, tP) in enumerate(tPiTest):
        tempTRow.append((VRs,tP))
    intensityWidthGrid.append(tempTRow)
VRs_grid, tP_grid = np.meshgrid(VRScan, tPiTest, indexing='ij')
intensityWidthGrid = np.stack((VRs_grid, tP_grid), axis=-1)
# (row, column)
# (VRs, timePulse)

intensityWidthGrid[:,0] # VRScan , tPiTest[0] 

intensityWidthGrid[0,:] # VRScan[0], tPiTest





ksz=kz
ksx=kx
VRScanOutput = []
VRScanOutputPi = [] 
VRScanOutputPM = []
VRScanOutputSC = []
multiPulses = []
VRScanTimeStart = datetime.now()
for (VRi,VRs) in enumerate(VRScan):
    VRScanTimeNow = datetime.now()
    tE = VRScanTimeNow - VRScanTimeStart
    if VRi != 0 :
        tR = (len(VRScan)-VRi)* tE/VRi
        l.info(f"Computing VRi={VRi}, VRs={round(VRs,2)}, tE {tE} tR {tR}")    
    tPiOutput = Parallel(n_jobs=N_JOBS)(
        delayed(lambda i: (i, scanTauPiInnerEval(i, False, False,0,p,0*dopd,VRs)[:2],ksz,ksx) )(i) 
        for i in tqdm(tPiTest)
    )   #### THIS THING TAKE A FEW MIN (or Hours)
    phiDensityGrid = np.zeros((len(tPiTest), pzlin.size))
    phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))
    
    # tPiOutputFramesDir = []
    output_prefix_tPiVscan = output_prefix+f"tPiScan ({VRi},{VRs})"
    os.makedirs(output_prefix_tPiVscan, exist_ok=True)
    # _ = Parallel(n_jobs=N_JOBS_PLT, timeout=1000)(
    #     delayed(tPiTestFrameExportHelper)(ti, tPiTest[ti], output_prefix_tPiVscan, skipMod=tPiTFEHSM)
    #         for ti in tqdm(range(len(tPiTest)))
    # )

    phiDensityGrid = np.zeros((len(tPiTest), pzlin.size))
    phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))
    # for i in tqdm(range(len(tPiTest))):
    for i in range(len(tPiTest)):
        item = tPiOutput[i]
        (swnf, phi) = phiAndSWNF(item[1][1])
        phiAbsSq = np.abs(phi)**2
        phiZ = np.trapz(phiAbsSq, pzlin,axis=0)
        phiDensityGrid[i] = phiZ
        for (j, hbar_k) in enumerate(hbar_k_transfers):
            index = pxlinIndexSet[j]
            phiDensityGrid_hbark[i,j] = np.trapz(phiZ[index], pzlin[index])

    VRScanOutput.append((VRi, phiDensityGrid, phiDensityGrid_hbark))
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark,axis=1)
    phiDensityNormed = phiDensityGrid_hbark / phiDensityNormFactor[:, np.newaxis]
    # indPDNPi = np.argmax(phiDensityNormed[:,hbarkInd])
    # indPDNPM = np.argmin(np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5))
    # VRScanOutputPi.append((indPDNPi, tPiTest[indPDNPi], phiDensityNormed[indPDNPi,hbarkInd]))
    # VRScanOutputPM.append((indPDNPM, tPiTest[indPDNPM], phiDensityNormed[indPDNPM,hbarkInd], phiDensityNormed[indPDNPM,hbarkInI]))
    # l.info(f"""indPDNPi = {indPDNPi} \ttPiTest[indPDNPi] = {round(tPiTest[indPDNPi]*1000,2)} μs \teff {round(phiDensityNormed[indPDNPi,hbarkInd],4)}
# indPDNPM = {indPDNPM} \ttPiTest[indPDNPM] = {round(tPiTest[indPDNPM]*1000,2)} μs \tef- {round(phiDensityNormed[indPDNPM,hbarkInd],4)} \tef+ {round(phiDensityNormed[indPDNPM,hbarkInI],4)}""")
    # SC_searchHelper=np.abs(phiDensityNormed[:,hbarkInA]-0.5)+np.abs(phiDensityNormed[:,hbarkInB]-0.5)
    # indPDNSC = np.argmin(SC_searchHelper)
    # VRScanOutputSC.append((indPDNSC, tPiTest[indPDNSC], phiDensityNormed[indPDNSC,hbarkInA], phiDensityNormed[indPDNSC,hbarkInB]))
    # l.info(f"""indPDNSC = {indPDNSC} \ttPiTest[indPDNSC] = {round(tPiTest[indPDNSC]*1000,3)} μs \tef-2 {round(phiDensityNormed[indPDNSC,hbarkInA],4)} \tef+2 {round(phiDensityNormed[indPDNSC,hbarkInB],4)}""")
    multiPulses.append(find_pulses(phiDensityNormed, thetaList))
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    nxm = int((nx-1)/2)
    nx2 = int((nx-1)/2)
    plt.imshow(phiDensityGrid, #phiDensityGrid[:,nxm-nx2:nxm+nx2], 
               extent=[pzlin[nxm-nx2]/(hb*k),pzlin[nxm+nx2]/(hb*k),0,len(tPiTest)], 
               interpolation='none',aspect=mom_dist_at_diff_angle_phi_asss)
    ax = plt.gca()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.ylabel("$dt =$"+str(dt*1000) + "$\mu \mathrm{s}$")
    plt.xlabel("$p_z \ (\hbar k)$")
    plt.subplot(1,2,2)
    plt.imshow(phiDensityGrid_hbark, 
               extent=[hbar_k_transfers[0],hbar_k_transfers[-1],0,len(tPiTest)], 
               interpolation='none',aspect=mom_dist_at_diff_angle_den_asss)
    plt.xlabel("$p_z \ (\hbar k)$ integrated block")
    
    title="mom_dist_at_diff_angle"
    plt.savefig(output_prefix_tPiVscan+"/"+title+".pdf", dpi=600)
    plt.savefig(output_prefix_tPiVscan+"/"+title+".png", dpi=600)
    plt.savefig(output_prefix_tPiVscan+" "+title+".png", dpi=600)
    # plt.show()
    plt.close()
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)
    plt.figure(figsize=(11,5))
    for (i, hbar_k) in enumerate(hbar_k_transfers):
        if abs(hbar_k) >5: continue
        if   hbar_k > 0: style = '+-'
        elif hbar_k < 0: style = 'x-'
        else:            style = '.-'
        plt.plot(tPiTest*1000, phiDensityGrid_hbark[:,i]/phiDensityNormFactor[i],
                 style, linewidth=1,alpha=0.5, markersize=5,
                 label=str(hbar_k)+"$\hbar k$",
                )
    
    plt.legend(loc=1,ncols=3)
    plt.ylabel("$normalised \int |\phi(p)| dp$ around region ($\pm$"+str(cut_p_width)+")")
    plt.xlabel("$t_\pi \ (\mu s)$")
    title = "bragg_pulse_duration_test_labeled"
    plt.savefig(output_prefix_tPiVscan+"/"+title+".pdf", dpi=600)
    plt.savefig(output_prefix_tPiVscan+"/"+title+".png", dpi=600)
    plt.savefig(output_prefix_tPiVscan+" "+title+".png", dpi=600)
    # plt.show()
    plt.close()
    
    indMax = np.argmax(phiDensityGrid_hbark[:,3]/phiDensityNormFactor[3])
    gc.collect()
l.info("""
==================================================================================
==================================================================================
====    ====    SCAN DONE    ====    =============================================
==================================================================================
==================================================================================
""")

with pgzip.open(output_prefix+'/_VRScanOutput'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(VRScanOutput, file) 
with pgzip.open(output_prefix+'/_intensityScan'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(intensityScan, file) 
with pgzip.open(output_prefix+'/_intensityWidthGrid'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(intensityWidthGrid, file) 
with pgzip.open(output_prefix+'/_multiPulses'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(multiPulses, file) 





# hbarkInd = 2
vtSliceM1 = np.empty((len(VRScan),len(tPiTest)))
for (VRi,VRs) in enumerate(VRScan):
    if VRi >= len(VRScanOutput): break
    phiDensityGrid_hbark = VRScanOutput[VRi][2]
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)
    phiDensityNormed = phiDensityGrid_hbark / phiDensityNormFactor[:, np.newaxis]
    # SC_searchHelper=np.abs(phiDensityNormed[:,hbarkInA]-0.5)+np.abs(phiDensityNormed[:,hbarkInB]-0.5)
    vtSliceM1[VRi] = phiDensityGrid_hbark[:,hbarkInd]/phiDensityNormFactor
    # vtSliceM1[VRi] = SC_searchHelper
print(hbar_k_transfers[hbarkInd])

vtSliceM1.shape

VRScanOutputPi

# +
# VRSOPi = np.array(VRScanOutputPi)
# VRSOPM = np.array(VRScanOutputPM)
# VRSOSC = np.array(VRScanOutputSC)
# -

pi04list = []
pi14list = [] 
pi24list = [] 
pi34list = []
pi44list = [] 
rowList = []
for (VRi, VRs) in enumerate(VRScan):
    this_result = multiPulses[VRi]
    peaks_ind = this_result[0]
    peaks_time = this_result[1]
    peaks_helper = this_result[2]
    # print(this_result)
    # break
    for (ind, ptime) in enumerate(peaks_time[0]): 
        pi04list.append((VRs,ptime,peaks_helper[0][ind]))
        rowList.append((VRs, VRs/VR/10, 0, peaks_ind[0][ind], ptime, peaks_helper[0][ind]))
    for (ind, ptime) in enumerate(peaks_time[1]): 
        pi14list.append((VRs,ptime,peaks_helper[1][ind]))
        rowList.append((VRs, VRs/VR/10, 1, peaks_ind[1][ind], ptime, peaks_helper[1][ind]))
    for (ind, ptime) in enumerate(peaks_time[2]): 
        pi24list.append((VRs,ptime,peaks_helper[2][ind]))
        rowList.append((VRs, VRs/VR/10, 2, peaks_ind[2][ind], ptime, peaks_helper[2][ind]))
    for (ind, ptime) in enumerate(peaks_time[3]): 
        pi34list.append((VRs,ptime,peaks_helper[3][ind]))
        rowList.append((VRs, VRs/VR/10, 3, peaks_ind[3][ind], ptime, peaks_helper[3][ind]))
    for (ind, ptime) in enumerate(peaks_time[4]): 
        pi44list.append((VRs,ptime,peaks_helper[4][ind]))
        rowList.append((VRs, VRs/VR/10, 4, peaks_ind[4][ind], ptime, peaks_helper[4][ind]))
pi04list = np.array(pi04list)
pi14list = np.array(pi14list)
pi24list = np.array(pi24list)
pi34list = np.array(pi34list)
pi44list = np.array(pi44list)

pulseOutput = pd.DataFrame(rowList, columns=["V","I","x","i","t","e"])

# + active=""
# this_result

# + active=""
# pi_lists = [[] for _ in range(len(thetaList))]
# for VRi, VRs in enumerate(VRScan):
#     peaks_ind, peaks_time, peaks_helper = multiPulses[VRi]
#     for i in range(5):
#         pi_lists[i].extend([(VRs, ptime, peaks_helper[i][ind]) for ind, ptime in enumerate(peaks_time[i])])    
# pi_arrays = [np.array(pi_list) for pi_list in pi_lists]
# pi04list, pi14list, pi24list, pi34list, pi44list = pi_arrays
# -

pi04list

pi04list[:,1]

# + active=""
# plt.scatter(pi04list[:,1], pi04list[:,0])
# -

np.linspace(tPiTest[-1],tPiTest[0],11)

# + editable=true slideshow={"slide_type": ""}
cmapS = plt.get_cmap('viridis', 20).copy()
cmapS.set_bad(color='black')
plt.close()
plt.figure(figsize=(12,8))
plt.imshow(np.fliplr(np.flipud(vtSliceM1)), 
           # aspect = 0.3,
           aspect=0.8*(tPiTest[0]-tPiTest[-1])/(intensityScan[-1]-intensityScan[0])*1000, 
           extent=[tPiTest[-1]*1000,(tPiTest[0]+tPDelta)*1000,intensityScan[0],intensityScan[-1]+isDelta],
           cmap=cmapS
          )
plt.colorbar(ticks=np.linspace(0,1,21))
plt.scatter(pi04list[:,1]*1000, pi04list[:,0]/VR/10+0.5*isDelta, color='red',marker='$0$',s=5,edgecolors='none')
plt.scatter(pi14list[:,1]*1000, pi14list[:,0]/VR/10+0.5*isDelta, color='red',marker='$1$',s=5,edgecolors='none')
plt.scatter(pi24list[:,1]*1000, pi24list[:,0]/VR/10+0.5*isDelta, color='red',marker='$2$',s=5,edgecolors='none')
plt.scatter(pi34list[:,1]*1000, pi34list[:,0]/VR/10+0.5*isDelta, color='red',marker='$3$',s=5,edgecolors='none')
plt.scatter(pi44list[:,1]*1000, pi44list[:,0]/VR/10+0.5*isDelta, color='red',marker='$4$',s=5,edgecolors='none')
# plt.scatter(VRSOPi[:,1]*1000,intensityScan+0.5*isDelta,color='red',marker='.',s=8)
# plt.scatter(VRSOPM[:,1]*1000,intensityScan+0.5*isDelta,color='fuchsia',marker='.',s=10)
# plt.scatter(VRSOSC[:,1]*1000,intensityScan+0.5*isDelta,color='fuchsia',marker='.',s=10)
plt.xlabel("Pulse width $\sigma$ $\mu s$")
plt.ylabel("Intensity $\mathrm{mW/mm^2}$")
# plt.xticks(ticks=np.round(np.linspace(0,200, 21)))  # Adding more x-ticks
# plt.yticks(ticks=np.linspace(intensityScan[0], intensityScan[-1] + isDelta, len(intensityScan) * 2))  # Adding more y-ticks
title = f"Transfer fraction of halo into {hbar_k_transfers[hbarkInd]}$\hbar k$ state"
plt.title(title)
# plt.savefig(output_prefix+"/"+title+".pdf", dpi=600)
# plt.savefig(output_prefix+"/"+title+".png", dpi=600)
plt.show()
# -



# + active=""
# rowList = []
# for (i,v) in enumerate(VRSOPi):
#     u = VRSOPM[i]
#     # print(f"{i}, {round(intensityScan[i],4)}, {round(VRScan[i],1)} \t"
#     #       +f"{v[0]}, {round(v[1],3)}, {round(v[2],4)} \t"
#     #       +f"{u[0]}, {round(u[1],3)}, {round(u[2],4)}, {round(u[3],4)}"
#     #  )
#     rowList.append((i,intensityScan[i],VRScan[i],v[0],v[1],v[2],u[0],u[1],u[2],u[3]))
# pulseOutput = pd.DataFrame(rowList, columns=["i","I","V","Pi","ts","ef","PM","ts","e-","e+"])

# + active=""
# rowList = []
# for (i,v) in enumerate(VRSOSC):
#     rowList.append((i,intensityScan[i],VRScan[i],v[0],v[1],v[2],v[3]))
# pulseOutput = pd.DataFrame(rowList, columns=["i","I","V","isc","tsc","e-","e+"])
# -

pulseOutput

pulseOutput.to_csv(output_prefix+"VRScanPulseDuration.csv")









# ### Loading previous scans

assert False, "just to catch run all"

# +
folder_to_import = '20240503-154711-TFF'
with pgzip.open(f'/Volumes/tonyNVME Gold/oneParticleSim/{folder_to_import}/VRScanOutput.pgz.pkl'
                , 'rb', thread=8) as file:
    VRScanOutput = pickle.load(file)

with pgzip.open(f'/Volumes/tonyNVME Gold/oneParticleSim/{folder_to_import}/intensityScan.pgz.pkl'
                , 'rb', thread=8) as file:
    intensityScan = pickle.load(file)

with pgzip.open(f'/Volumes/tonyNVME Gold/oneParticleSim/{folder_to_import}/intensityWidthGrid.pgz.pkl'
                , 'rb', thread=8) as file:
    intensityWidthGrid = pickle.load(file)
# -

intensityScan = intensityScan[:88]

VRScan = intensityWidthGrid[:,0][:,0][:88]

isDelta = VRScan[1]-VRScan[0]

tPiTest = intensityWidthGrid[0,:][:,1]

VRScanOutputPi = [] 
VRScanOutputPM = []
intensityScan2 = []
for (VRi,VRs) in enumerate(VRScan):
    phiDensityGrid = VRScanOutput[VRi][1]
    phiDensityGrid_hbark = VRScanOutput[VRi][2]
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark,axis=1)
    phiDensityNormed = phiDensityGrid_hbark / phiDensityNormFactor[:, np.newaxis]
    indPDNPi = np.argmax(phiDensityNormed[:,hbarkInd])
    indPDNPM = np.argmin(np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5))
    VRScanOutputPi.append((indPDNPi, tPiTest[indPDNPi], phiDensityNormed[indPDNPi,hbarkInd]))
    VRScanOutputPM.append((indPDNPM, tPiTest[indPDNPM], phiDensityNormed[indPDNPM,hbarkInd], phiDensityNormed[indPDNPM,hbarkInI]))
    intensityScan2.append(intensityScan[VRi])

intensityScan2 = np.array(intensityScan2)

np.shape(VRScanOutputPi)

np.shape(intensityScan2)



VRSOPi.shape 

intensityScan.shape







# ## Detuning Scan

haloId = np.arange(-9,9+2,2)
l.info(f"haloId = {haloId} \nNote pzmax/p = {pzmax/p}")
assert max(haloId)+1.5 < pzmax/p, "check halo max momentum"


# +
def momAngMaskCombov2(mA, pwid=5*dpz, hid=0):
    #NOTE: this should supercede momAngMaskCombo
    #REVIEW: this thing might contain a +/- p definition error, or could mean somewhere in the code is problematic
    momAngMaskL = np.exp(-((pxlin[:,np.newaxis] + p*sin(mA))**2 + (-pzlin[np.newaxis,:] - hid*p + p*cos(mA))**2) / (pwid)**2)
    momAngMaskR = np.exp(-((pxlin[:,np.newaxis] - p*sin(mA))**2 + (-pzlin[np.newaxis,:] - hid*p - p*cos(mA))**2) / (pwid)**2)
    # momAngMaskDR = np.exp(-((pxlin[:,np.newaxis] - p*sin(mA))**2 + (-pzlin[np.newaxis,:] + p - p*cos(mA))**2) / (pwid)**2)
    # momAngMaskDL = np.exp(-((pxlin[:,np.newaxis] + p*sin(mA))**2 + (-pzlin[np.newaxis,:] + p + p*cos(mA))**2) / (pwid)**2)
    return (momAngMaskL, momAngMaskR)

def momAngFValv2(mA,phi,pwid=5*dpz):
    #NOTE: this should supercede momAngFVal
    outputTemp = np.zeros((len(haloId), 2))
    phiAbsSq = np.abs(phi)**2
    for (i,h) in enumerate(haloId):
        mAML, mAMR = momAngMaskCombov2(mA, pwid, h)
        outputTemp[i,0] = np.sum(phiAbsSq * mAML * dpz * dpx)
        outputTemp[i,1] = np.sum(phiAbsSq * mAMR * dpz * dpx)
    return outputTemp



# -

ddS = 1/8
deltaScan = np.arange(-2 ,2+ddS,ddS)
deltaScanRU = deltaScan*dopd
l.info(f"""len(deltaScan) = {len(deltaScan)}, ddS = {ddS}
deltaScan = {round(deltaScan[0],6)}, {round(deltaScan[1],6)}, ..., {round(deltaScan[-2],6)}, {round(deltaScan[-1],6)}
deltaScanRU = {round(deltaScanRU[0],2)}, {round(deltaScanRU[1],2)}, ..., {round(deltaScanRU[-2],2)}, {round(deltaScanRU[-1],2)}
dopd = {dopd}""")
l.info(f"roughtly can finish in {round((tPiScanTime10usDelta*tPiScanTotalSimMS*100*len(deltaScan)/(0.7*N_JOB2)).total_seconds()/60/60, 2)} hours")

momAngList = np.arange(0,180,1)*pi/180
DSScanOutput = np.zeros((len(deltaScan), len(tPiTest), len(momAngList), len(haloId), 2))
l.info(f"""DSScanOutput.shape = {DSScanOutput.shape}, size {DSScanOutput.size}, {round(sys.getsizeof(DSScanOutput)/1024**2,3)} MB""")

# with pgzip.open(output_prefix+"haloId"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
#     pickle.dump(haloId, file)
# with pgzip.open(output_prefix+"deltaScan"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
#     pickle.dump(deltaScan, file)
# with pgzip.open(output_prefix+"tPiTest"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
#     pickle.dump(tPiTest, file)
# with pgzip.open(output_prefix+"momAngList"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
#     pickle.dump(momAngList, file)
with pgzip.open(output_prefix+"DSScanConfigs"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump((deltaScan, tPiTest, momAngList, haloId), file)

max_length = max(len(deltaScan), len(tPiTest), len(momAngList), len(haloId))
df = pd.DataFrame({
    'index': np.arange(max_length),
    'deltaScan':    np.pad(np.round(deltaScan,7),   (0, max_length - len(deltaScan)),   'constant', constant_values=np.nan),
    'tPiTest':      np.pad(np.round(tPiTest,7),     (0, max_length - len(tPiTest)),      'constant', constant_values=np.nan),
    'momAngList':   np.pad(np.round(momAngList*180/pi,6),  (0, max_length - len(momAngList)),  'constant', constant_values=np.nan),
    'haloId':       np.pad(haloId*1.0,  (0, max_length - len(haloId)),      'constant', constant_values=np.nan), 
})
df.to_csv(output_prefix+'DSScanConfigs.csv', index=False)

# +
# for (i,v) in enumerate(deltaScan): print(f"{i}: {round(v,6)}")
# for (i,v) in enumerate(tPiTest): print(f"{i}: {round(v,6)}")
# for (i,v) in enumerate(momAngList): print(f"{i}: {round(v,6)}")
# for (i,v) in enumerate(haloId): print(f"{i}: {round(v,6)}")

# +
DScanTimeStart = datetime.now()
output_prefix_dScanLoopTemp = output_prefix+"dScanLoopTemp/"
os.makedirs(output_prefix_dScanLoopTemp, exist_ok=True)
for (di, dd) in enumerate(deltaScan):
    DScanTimeNow = datetime.now()
    tE = DScanTimeNow - DScanTimeStart
    if di != 0 :
        tR = (len(deltaScan)-di)* tE/di
        l.info(f"Computing di={di}, dd={round(dd,2)}, tE {tE} tR {tR}")
    else:
        l.info(f"Computing di={di}, dd={round(dd,2)}")

    tPiOutput = Parallel(n_jobs=N_JOB2)(
        delayed(lambda i: (i, scanTauPiInnerEval(i, False, False, 0, p, dd*dopd,VR)[:2]) )(i) 
        for i in tqdm(tPiTest)
    )
    
    gc.collect()

    with pgzip.open(output_prefix_dScanLoopTemp+f"tPiOutput_di={di}"+output_ext, 'wb', thread=5, blocksize=1*10**8) as file:
        pickle.dump(tPiOutput, file)

    momAngPiScan = np.zeros((len(tPiTest), len(momAngList), len(haloId), 2))
    for (ti, tt) in tqdm(enumerate(tPiTest), total=len(tPiTest)):
        item = tPiOutput[ti]
        (swnf, phi) = phiAndSWNF(item[1][1])
        momAngResults = Parallel(n_jobs=N_JOBS)( 
            delayed(lambda mAv: np.array(momAngFValv2(mAv, pwid=5*dpz, phi=phi)))(mAv)
            for mAv in momAngList
        )
        momAngResults = np.array(momAngResults)
        momAngPiScan[ti] = momAngResults
    DSScanOutput[di]=momAngPiScan

    with pgzip.open(output_prefix_dScanLoopTemp+f"momAngPiScan_di={di}"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
        pickle.dump(momAngPiScan, file)

    del momAngPiScan, tPiOutput
    gc.collect()

with pgzip.open(output_prefix+"DSScanOutput"+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(DSScanOutput, file)
l.info("Detuning Scan DONE !!! YEAHHH ")
# -

DSScanOutput.shape

output_prefix_dScanLoopFig = output_prefix+"dScanLoopFig/"
os.makedirs(output_prefix_dScanLoopFig, exist_ok=True)

plt.plot(DSScanOutput[1,1,:,4,0],'-')
plt.plot(DSScanOutput[1,1,:,4,1],'-')
plt.plot(DSScanOutput[1,1,:,5,0],'-')
plt.plot(DSScanOutput[1,1,:,5,1],'-')
# plt.plot(DSScanOutput[1,2,:,4,0],'.-')
# plt.plot(DSScanOutput[1,2,:,4,1],'.-')
plt.show()


def fig_tScan_at(dID=16, aID=90, logging=False, saveFig=False, showFig=False):
    normalisation_check = np.sum(DSScanOutput[dID,:,aID,:,0],axis=1)
    normalisation = np.mean(normalisation_check)
    normalisationSTD = np.std(normalisation_check)
    if logging: l.info(f"""Normalisation check: {normalisation:.8f} ± {normalisationSTD:.8f}
Norm uncertainty {normalisationSTD/normalisation*100:.4f}%, max {np.max(normalisation_check/normalisation):.6f}, min {np.min(normalisation_check/normalisation):.6f}""")

    for (hi, hp) in enumerate(haloId):
        plt.plot(tPiTest*1000,DSScanOutput[dID,:,aID,hi,0]/normalisation,'-',label=f"${'-' if hp<0 else '+'} {abs(hp)}$", alpha=0.7)
        # normalisation_check += DSScanOutput[dID,:,90,hi,0]

    # plt.plot(tPiTest,DSScanOutput[dID,:,90,5,0],'-')

    # plt.plot(tPiTest,normalisation_check,'--',label="$\Sigma$", color='gray')

    plt.legend(loc='center right',title="Halo center $p$", fontsize=8, ncol=2)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1/2))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.1))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1/4))
    
    plt.ylim(-0.025,1.025)
    plt.xlim(-0.5, 40.5)

    plt.grid(which='major', linestyle='-', linewidth='0.3', color='black', alpha=0.4)
    plt.grid(which='minor', linestyle='-', linewidth='0.3', color='black', alpha=0.1)

    plt.xlabel("Pulse width $(\mu s)$")
    plt.ylabel(f"Population Fraction \n(Normalisation Uncertainty {normalisationSTD/normalisation*100:.4f}%)")
    # title="DSScanOutput_dID,|,aID,|,0"
    title=f"Detuning={deltaScan[dID]:.3f}$\omega_\delta$, Angle={round(momAngList[aID]*180/pi,6)}$^\circ$"
    plt.title(title)
    title_clean = re.sub(r'\$.*?\$', '', title)
    if logging: l.info(f"Export file name: {title_clean}")
    if saveFig:
        plt.savefig(output_prefix_dScanLoopFig+title_clean+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_prefix_dScanLoopFig+title_clean+".png", dpi=600, bbox_inches='tight')
    if showFig: plt.show() 
    plt.close()
    return(dID, aID, title_clean)


fig_tScan_at(dID=16, aID=90, logging=True, saveFig=True, showFig=True)

# dScanLoopFigNames = []
# for dID in range(len(deltaScan)):
#     for aID in range(len(momAngList)):
#         dScanLoopFigNames.append(fig_tScan_at(dID, aID, logging=False, saveFig=True))
dScanLoopFigNames = Parallel(n_jobs=N_JOB2)(
    delayed(lambda dID, aID: fig_tScan_at(dID, aID, logging=False, saveFig=True, showFig=False))(dID, aID)
    for aID in tqdm(range(len(momAngList)))
    for dID in range(len(deltaScan)) 
)

len(dScanLoopFigNames)

dSLFNInd = [[None for _ in range(len(momAngList))] for _ in range(len(deltaScan))]
for index, (a, b, _) in enumerate(dScanLoopFigNames):
    dSLFNInd[a][b] = index
dSLFNInd = np.array(dSLFNInd)

dScanLoopFigNames[dSLFNInd[16,90]]

output_prefix_dScanLoopMov = output_prefix+"dScanLoopMov/"
os.makedirs(output_prefix_dScanLoopMov, exist_ok=True)

for aID in tqdm(range(len(momAngList))):
    dSLFN_pngs = [output_prefix_dScanLoopFig+dScanLoopFigNames[i][2]+".png" for i in dSLFNInd[:,aID]]
    img_alt_frames = Parallel(n_jobs=-1)(
        delayed(lambda image: cv2.imread(image))(image) 
        # for image in tqdm(dSLFN_pngs, desc="Processing Images")
        for image in dSLFN_pngs
    )

    if img_alt_frames:  # Determine the width and height from the first image if not empty
        height, width, layers = img_alt_frames[0].shape # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')  # HEVC codec
        out = cv2.VideoWriter(
            output_prefix_dScanLoopMov+f"Angle={round(momAngList[aID]*180/pi,6)}"+".mov", 
            fourcc, 10.0, (width, height), True) # Write img_alt_frames to the video file
        # for frame in tqdm(img_alt_frames, desc="Writing Video"):
        for frame in img_alt_frames:
            out.write(frame)  # Write out frame to video
        out.release()  # Release the video writer
        del frame, out
    else:
        print("No images found or processed.")
    del img_alt_frames
    gc.collect()


for dID in tqdm(range(len(deltaScan))):
    dSLFN_pngs = [output_prefix_dScanLoopFig+dScanLoopFigNames[i][2]+".png" for i in dSLFNInd[dID,:]]
    img_alt_frames = Parallel(n_jobs=-1)(
        delayed(lambda image: cv2.imread(image))(image) 
        # for image in tqdm(dSLFN_pngs, desc="Processing Images")
        for image in dSLFN_pngs
    )
    if img_alt_frames:
        height, width, layers = img_alt_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')
        out = cv2.VideoWriter(
            output_prefix_dScanLoopMov+f"Detuning={deltaScan[dID]:.3f}"+".mov", 
            fourcc, 10.0, (width, height), True)
        for frame in img_alt_frames:
            out.write(frame)
        del frame, out
    else: print("No images found or processed.")
    del img_alt_frames
    gc.collect()




# +
X, Y = np.meshgrid(tPiTest*1000, momAngList*180/pi)
Z = DSScanOutput[16,:,:,4,0].T

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0.5, alpha=0.3, rstride=10, cstride=10, edgecolors='royalblue')
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=180, cmap='coolwarm')

plt.show()
# -

norm2d_check = np.sum(DSScanOutput[16,:,:,:,0],axis=2)
plt.imshow((norm2d_check.T)[40:-40,:], aspect='auto', interpolation='none', cmap='jet', 
    extent=[tPiTest[-1]*1000,tPiTest[0]*1000,momAngList[40]*180/pi,momAngList[-40]*180/pi])
plt.colorbar()
plt.show()

# +
# plt.imshow(DSScanOutput[16,::-1,:,4,0].T, 
#     extent=[tPiTest[-1]*1000,tPiTest[0]*1000,momAngList[0]*180/pi,momAngList[-1]*180/pi], 
#     aspect='auto', interpolation='none', cmap='jet')

X, Y = np.meshgrid(tPiTest*1000, momAngList*180/pi)
Z = DSScanOutput[16,:,:,4,0].T
plt.contourf(X, Y, Z, 60, cmap='jet')
# plt.contourf(X, Y, norm2d_check.T, 60, cmap='jet')

plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=15))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
plt.grid(which='major', linestyle='-', linewidth='0.2', color='white', alpha=1)
plt.grid(which='minor', linestyle='-', linewidth='0.15', color='white', alpha=1)
# plt.colorbar(label="Probability $|⟨\star|\psi⟩|^2$", ticks=matplotlib.ticker.MaxNLocator(nbins=20))
cbar = plt.colorbar(label="(× $10^{-2}$) Probability $|⟨\star|\psi⟩|^2$", ticks=matplotlib.ticker.MaxNLocator(nbins=10))
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x * 100:.2f}'))
plt.xlabel("Bragg pulse width duration ($\mu s$)")
plt.ylabel("Polar Angle (deg) from halo north pole")

plt.show()

# +
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(tPiTest * 1000, momAngList * 180 / np.pi)

# Plot each of the 10 slices in the 3D plot
for (hii, hid) in enumerate(haloId):
    if abs(hid) > 4: continue
    Z = DSScanOutput[16, :, :, hii, 0].T
    ax.contourf(X, Y, Z, 60, zdir='z', offset=hid, cmap=ct_cmap('Blues'))

# Set labels and title
ax.set_xlabel('Bragg pulse width duration ($\mu s$)')
ax.set_ylabel('Polar Angle (deg) from halo north pole')
ax.set_zlabel('Slice Index')
ax.set_title('3D Contour Plots of 10 Slices')
ax.set_zlim(4, -4)
ax.view_init(elev=20, azim=55)
ax.set_box_aspect(aspect = (1,1,2))
plt.show()
# -

haloId[3:-3]

output_prefix_dScan2DFig = output_prefix+"dScan2DFig/"
os.makedirs(output_prefix_dScan2DFig, exist_ok=True)



def fig_dScan2D_at(dID=16, haloIdCut=4, logging=False, saveFig=False, showFig=False, figWidcm=20):
    haloIdStriped = range(len(haloId))[haloIdCut:-haloIdCut]
    if logging: l.info(f"haloIdStriped = {haloIdStriped}, \t {haloId[haloIdStriped]}")
    dV = deltaScan[dID]
    # dID = 16
    fig, axes = plt.subplots(1, len(haloIdStriped), figsize=(figWidcm*cm, 10*cm), sharey=True, constrained_layout=True)
    X, Y = np.meshgrid(tPiTest*1000, momAngList*180/pi)
    vmin = np.min(DSScanOutput[dID, :, :, haloIdStriped, 0])
    vmax = np.max(DSScanOutput[dID, :, :, haloIdStriped, 0])
    for (hi, hp) in enumerate(haloIdStriped):
        Z = DSScanOutput[dID,:,:,hp,0].T
        ax = axes[hi]
        contour = ax.contourf(X, Y, Z, 60, cmap='jet', vmin=vmin, vmax=vmax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=15))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
        ax.grid(which='major', linestyle='-', linewidth='0.2', color='white', alpha=1)
        ax.grid(which='minor', linestyle='-', linewidth='0.15', color='white', alpha=1)
        ax.set_xlabel("Bragg pulse width duration ($\mu s$)")
        # ax.set_ylabel("Polar Angle (deg) from halo north pole")
        ax.set_title(f"Halo ${'-' if haloId[hp]<0 else '+'} {abs(haloId[hp])}$")
    axes[0].set_ylabel("Polar Angle (deg) from halo north pole")

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    fig.colorbar(contour, ax=axes, cax=cax, label="(× $10^{-2}$) Probability $|⟨\star|\psi⟩|^2$", 
        ticks=matplotlib.ticker.MaxNLocator(nbins=10)).ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x * 100:.2f}'))
    title=f"Detuning={dV:.3f}$\omega_\delta$"
    fig.suptitle(title)
    title_clean = re.sub(r'\$.*?\$', '', title)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=1, hspace=0)
    if saveFig:
        plt.savefig(output_prefix_dScan2DFig+title_clean+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_prefix_dScan2DFig+title_clean+".png", dpi=600, bbox_inches='tight')
    if showFig: plt.show()
    plt.close()
    return (dID, title_clean)


# fig_dScan2D_at(dID=16, haloIdCut=4, logging=True, saveFig=False, showFig=True)
fig_dScan2D_at(dID=16, haloIdCut=3, logging=True, saveFig=False, showFig=True,figWidcm=30)
# fig_dScan2D_at(dID=16, haloIdCut=2, logging=True, saveFig=False, showFig=True)

# for dID in tqdm(range(len(deltaScan)), desc="Exporting Detuning Scans"):
#     fig_dScan2D_at(dID, haloIdCut=3, logging=False, saveFig=True, showFig=False, figWidcm=30)
dScan2DFigNames = Parallel(n_jobs=N_JOB2)(
    delayed(lambda dID: fig_dScan2D_at(dID, haloIdCut=3, logging=False, saveFig=True, showFig=False, figWidcm=30))(dID)
    for dID in tqdm(range(len(deltaScan)))
)

output_prefix_dScan2DMov = output_prefix+"dScan2DMov/"
os.makedirs(output_prefix_dScan2DMov, exist_ok=True)

# +
# for dID in tqdm(range(len(deltaScan)), desc="Exporting Detuning Scans"):
dS2FN_pngs = [output_prefix_dScan2DFig+iii[1]+".png" for iii in dScan2DFigNames]
img_alt_frames = Parallel(n_jobs=-1)(
    delayed(lambda image: cv2.imread(image))(image) 
    for image in tqdm(dS2FN_pngs)
)
if img_alt_frames:
    height, width, layers = img_alt_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'hvc1')
    out = cv2.VideoWriter(
        output_prefix_dScan2DMov+f"Halo Transfers varying different detuning"+".mov", 
        fourcc, 10.0, (width, height), True)
    for frame in tqdm(img_alt_frames):
        out.write(frame)
    del frame, out  
    del img_alt_frames

gc.collect()
# -






