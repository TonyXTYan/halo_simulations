#!/usr/bin/env python
# coding: utf-8

# # One Particle - High Resolution

# In[1]:


# get_ipython().system('python -V')


# In[2]:


from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from math import *
from uncertainties import *
from scipy.stats import chi2
import scipy
from matplotlib import gridspec
import matplotlib
import pandas
import sys
import statsmodels.api as sm
import warnings ## statsmodels.api is too old ... -_-#

import pickle
import pgzip
import os
import platform
import logging
import sys

N_JOBS=6
# from tqdm.notebook import tqdm
from tqdm import tqdm
from datetime import datetime
# from numba import jit, njit
# import numba

import pyfftw
nthreads=2

# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots


# In[131]:


import gc
gc.enable(  )


# In[3]:


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


# In[4]:


plt.set_loglevel("warning")


# In[5]:


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


# In[6]:


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


# In[7]:


l.info(f"""nthreads = {nthreads}
N_JOBS = {N_JOBS}""")


# In[8]:


# np.show_config()


# In[9]:


nx = 1000+1
nz = 1000+1
xmax = 50 #Micrometers
# zmax = (nz/nx)*xmax
zmax = 50
dt = 0.1e-3 # Milliseconds
dx = 2*xmax/(nx-1)
dz = 2*zmax/(nz-1)
hb = 63.5078 #("AtomicMassUnit" ("Micrometers")^2)/("Milliseconds")
m3 = 3   # AtomicMassUnit
m4 = 4 

pxmax = 2*pi*hb/dx/2
pzmax = 2*pi*hb/dz/2
# pxmax= (nx+1)/2 * 2*pi/(2*xmax)*hb # want this to be greater than p
# pzmax= (nz+1)/2 * 2*pi/(2*zmax)*hb


# In[10]:


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


# In[11]:


l.info(f"""rotate phase per dt for m3 = {1j*hb*dt/(2*m3*dx*dz)} \t #want this to be small
rotate phase per dt for m4 = {1j*hb*dt/(2*m4*dx*dz)} 
number of grid points = {round(nx*nz/1000/1000,3)} (million)
minutes per grid op = {round((nx*nz)*0.001*0.001/60, 3)} \t(for 1μs/element_op)
""")


# In[12]:


wavelength = 1.083 #Micrometers
beam_angle = 90
k = sin(beam_angle*pi/180/2) * 2*pi / wavelength # effective wavelength
kx = 0
kz = k

#alternatively 
# k = pi / (4*dx)
# beam_angle = np.arcsin(k/(2*pi/wavelength))*180/pi

# print("k =", k, " is 45 degree Bragg")
# k = 2*pi / wavelength
# k = pi / (2*dz)
# k = pi / (4*dx)
# dopd = 60.1025 # 1/ms Doppler detuning (?)

p = hb*k
# print("k  =",k,"1/µm")
# print("p  =",p, "u*µm/ms")
v4 = hb*k/m4
v3 = hb*k/m3
# print("v3 =",v3, "µm/ms")
# print("v4 =",v4, "µm/ms")

# sanity check
# assert (pxmax > p*2.5 or pzmax > p*2.5), "momentum resolution too small"
# dopd = 60.1025 # 1/ms Doppler detuning (?)
dopd = v3**2 * m3 / hb


# In[13]:


l.info(f"""wavelength = {wavelength} µm
beam_angle = {beam_angle}
k = {k} 1/µm
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


# In[14]:


l.info(f"""Maximum simulation time {xmax/v3}, which is steps {xmax/v3/dt}""")


# In[15]:


pxmax**2/m3*dt/hb


# In[16]:


-(1j/hb) * (0.5/m3) * (dt)*pxmax


# In[17]:


dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb
pxlin = np.linspace(-pxmax,+pxmax,nx)
pzlin = np.linspace(-pzmax,+pzmax,nz)
# print("(dpx,dpz) = ", (dpx, dpz))
if abs(dpx - (pxlin[1]-pxlin[0])) > 0.0001: l.error("AHHHHH px")
if abs(dpz - (pzlin[1]-pzlin[0])) > 0.0001: l.error("AHHHHH pz")
l.info(f"""dpx = {dpx} uµm/ms
dpz = {dpz} """)


# In[18]:


#### WARNING:
###  These frequencies are in Hz, 
#### This simulation uses time in ms, 1Hz = 0.001 /ms
a4 = 0.007512 # scattering length µm
intensity1 = 1 # mW/mm^2 of beam 1
intensity2 = 1 
intenSat = 0.0017 # mW/mm^2
linewidth = 2*pi*1.6e6 # rad * Hz
omega1 = linewidth * sqrt(intensity1/intenSat/2)
omega2 = linewidth * sqrt(intensity2/intenSat/2)
detuning = 2*pi*3e9 # rad*Hz
omegaRabi = omega1*omega2/2/detuning # rad/s

VR = 2*hb*omegaRabi*0.001 # Bragg lattice amplitude # USE THIS ONE! 

omega = 50 # two photon Rabi frequency # https://doi.org/10.1038/s41598-020-78859-1
V0 = 2*hb*omega # Bragg lattice amplitude

# tBraggPi = np.sqrt(2*pi*hb)/V0 
tBraggPi = 2*pi/omegaRabi*1000 
tBraggCenter = tBraggPi * 5
tBraggEnd = tBraggPi * 10

V0F = 50*1000


# In[19]:


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


# In[20]:


def V(t):
    return V0 * (2*pi)**-0.5 * tBraggPi**-1 * np.exp(-0.5*(t-tBraggCenter)**2 * tBraggPi**-2)

def VB(t, tauMid, tauPi):
    return V0 * (2*pi)**-0.5 * tauPi**-1 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)


# @jit
# @jit(cache=True, nopython=True)
# @jit(forceobj=True, cache=True)
def VBF(t, tauMid, tauPi, V0FArg=V0F):
    return V0FArg * (2*pi)**-0.5 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)


# In[21]:


l.info(f"term infront of Bragg potential {1j*(dt/hb)}")
l.info(f"max(V)  {1j*(dt/hb)*V(tBraggCenter)}")
l.info(f"max(VR) {1j*(dt/hb)*VBF(tBraggCenter,tBraggPi,VR)}")


# In[22]:


xlin = np.linspace(-xmax,+xmax, nx)
zlin = np.linspace(-zmax,+zmax, nz)
psi=np.zeros((nx,nz),dtype=complex)
zones = np.ones(nz)
xgrid = np.tensordot(xlin,zones,axes=0)
# cosGrid = np.cos(2*k*xgrid)
# cosGrid = np.zeros((nx,nz))
# for (ix,x) in enumerate(xlin):
#     cosGrid[ix,:] = np.cos(2*kx*x + 2*kz*zlin)
cosGrid = np.cos(2 * kx * xlin[:, np.newaxis] + 2 * kz * zlin)


# In[23]:


if abs(dx - (xlin[1]-xlin[0])) > 0.0001: l.error("AHHHHx")
if abs(dz - (zlin[1]-zlin[0])) > 0.0001: l.error("AHHHHz")


# In[24]:


l.info(f"{round(psi.nbytes/1000/1000 ,3)} MB of RAM for psi")


# In[25]:


tbtest = np.arange(tBraggCenter-5*tBraggPi,tBraggCenter+5*tBraggPi,dt)
plt.plot(tbtest, VBF(tbtest,tBraggPi*5,tBraggPi))
plt.plot(tbtest, VBF(tbtest,tBraggPi*5,tBraggPi,VR))
l.info(f"max(V) {1j*(dt/hb)*VBF(tBraggCenter,tBraggPi*5,tBraggPi)}")


# In[26]:


V(tBraggCenter)


# In[27]:


VBF(tBraggCenter,tBraggPi*5,tBraggPi)


# In[28]:


np.trapz(V(tbtest),tbtest) # this should be V0


# In[29]:


ncrop = 30
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cosGrid.T,aspect=1)
plt.title("bragg potential grid smooth?")

plt.subplot(2,2,2)
plt.imshow(cosGrid[:ncrop,:ncrop].T,aspect=1)
plt.title("grid zoomed in")

plt.subplot(2,2,3)
# plt.plot(cosGrid[:,0],alpha=0.9,linewidth=0.1)
plt.plot(cosGrid[0,:],alpha=0.9,linewidth=0.1)

plt.subplot(2,2,4)
# plt.plot(cosGrid[:ncrop,0],alpha=0.9,linewidth=0.5)
plt.plot(cosGrid[0,:ncrop],alpha=0.9,linewidth=0.5)

title="bragg_potential_grid"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show(block=False)


# In[ ]:





# In[30]:


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
    
    if plt_show: plt.show(block=False)


# In[31]:


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
#     plt.show(block=False)
    
    if plt_show: plt.show(block=False)
        


# In[32]:


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


# In[33]:


# @njit(parallel=True)
# @jit(nopython=True) 
# @jit(forceobj=True)
# @jit
# @jit(cache=True)
# @jit(cache=False)
def psi0ringUnNormOffset(x,z,pr=p,mur=10,sg=sg,xo=0,zo=0,pxo=0,pzo=0):
    return 1 \
            * np.exp(-0.5*( mur - np.sqrt((x-xo)**2 + (z-zo)**2) )**2 / sg**2) \
            * np.exp(+(1j/hb) * (((x-xo)**2 + (z-zo)**2)**0.5 * pr + x*pxo+z*pzo))


# In[34]:


# V00 = 50000
# dt=0.01
# VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * V00 * cosGrid )

expPGrid = np.zeros((nx,nz),dtype=complex)
for indx in range(nx):
    expPGrid[indx, :] = np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin[indx]**2 + pzlin**2))  


# In[35]:


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


# In[36]:


# @jit(forceobj=True)
# @njit(forceobj=True)
# @jit(cache=True)
def psi0ringNpOffset(mur=1,sg=1,pr=p,xo=0,zo=0,pxo=0,pzo=0):
    psi = np.zeros((nx,nz),dtype=np.complex128)
    for ix in range(1,nx-1):
        x = xlin[ix]
        psi[ix][1:-1] = psi0ringUnNormOffset(x,zlin[1:-1],pr,mur,sg,xo,zo,pxo,pzo)
    norm = np.sum(np.abs(psi)**2)*dx*dz
    psi *= 1/sqrt(norm)
    return psi


# In[37]:


# @jit(forceobj=True, cache=True)
# @jit(forceobj=True)
# @jit('(complex128[:,:])', forceobj=True, cache=True)
# @jit
# @njit
def phiAndSWNF(psi):
    phiUN = np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))
    # superWeirdNormalisationFactorSq = np.trapz(np.trapz(np.abs(phiUN)**2, pxlin, axis=0), pzlin)
    superWeirdNormalisationFactorSq = np.sum(np.abs(phiUN)**2)*dpx*dpz
    swnf = sqrt(superWeirdNormalisationFactorSq)
    phi = phiUN/swnf
    return (swnf, phi)


# In[38]:


# psi = psi0np(5,5,0,0)
# psi = psi0np(5,5,-0.5*p,0)
# psi = psi0np(1,1,p,p)
# psi = psi0np(1,1,0,0)


    
# psi = psi0ringNp(10,1.7,1*hb*k)
psi = psi0ringNpOffset(5,1,p,0,5,0,p)
# psi = psi0np(2,2,0.5*p*np.cos(0),0.5*p*np.sin(0))
(swnf, phi) = phiAndSWNF(psi)

t = 0

print("Super weird normalisation factor, swnf =",swnf)
print(np.sum(np.abs(phi)**2)*dpx*dpz, "|phi|**2 normalisation check")
print(np.sum(np.abs(psi)**2)*dx*dz,   "|psi|**2 normalisation check")
plot_psi(psi,False)
title="init_ring_psi"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show(block=False)

plot_mom(psi,5,5,False)
title="init_ring_phi"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show(block=False)


# In[ ]:





# In[39]:


# @jit(nopython=False) 
# @jit(forceobj=True, cache=True)
# @jit
def toMomentum(psi, swnf) -> np.ndarray:
    return np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))/swnf
# @jit(nopython=False) 
# @jit(forceobj=True, cache=True)
# @jit
def toPosition(phi, swnf) -> np.ndarray:
    return pyfftw.interfaces.numpy_fft.ifft2(np.fft.ifftshift(np.fliplr(phi*swnf)),threads=nthreads,norm='ortho')


# In[40]:


def plotNow(t, psi):
        print("time =", round(t*1000,4), "µs")
        print(np.sum(np.abs(psi)**2)*dx*dz,"|psi|^2")
        print(np.sum(np.abs(phi)**2)*dpx*dpz,"|phi|^2")
        plot_psi(psi)
        plot_mom(psi)


# In[41]:


# @jit
# @jit(forceobj=True, cache=True)
def loop(t,psi,tauPi,tauMid,V0FArg,doppd,phase,swnf,kkx,kkz):
    # nonlocal V0FArg
    # cosGrid = np.cos(2*k*xgrid + doppd*(t-tBraggCenter) + phase)
    cosGrid = np.cos(2*kkx*xlin[:,np.newaxis] + 2*kkz*zlin + doppd*(t-tauMid) + phase)
    VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,V0FArg) * cosGrid )
    psi *= VxExpGrid
    phi = toMomentum(psi,swnf)
    phi *= expPGrid
    psi = toPosition(phi,swnf)
    psi *= VxExpGrid
    
    # if print_every_t > 0 and step % round(print_every_t / dt) == 0: 
    #     plotNow(t,psi)
    t += dt 
    return (t,psi)


# In[42]:


_ = loop(0,psi0ringNpOffset(10,1.7,p,0,+10,0,p),1,1,1,0,0,1,kx,kz)


# In[43]:


# @njit(parallel=True)
# @jit(nopython=True) 
# @jit(nopython=False) 
# @njit
# @jit(forceobj=True, cache=True)
# @jit(forceobj=True, nopython=False)
# @jit
# @jit(forceobj=True, parallel=True)
# @jit(parallel=True)
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

    # @jit(forceobj=True)
    # @jit
    # def loop():
    #     nonlocal t
    #     nonlocal psi
    #     nonlocal phi
    #     # nonlocal V0FArg
    #     # cosGrid = np.cos(2*k*xgrid + doppd*(t-tBraggCenter) + phase)
    #     cosGrid = np.cos(2*kx*xlin[:,np.newaxis] + 2*kz*zlin + doppd*(t-tBraggCenter) + phase)
    #     VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,V0FArg) * cosGrid )
    #     psi *= VxExpGrid
    #     phi = toMomentum(psi,swnf)
    #     phi *= expPGrid
    #     psi = toPosition(phi,swnf)
    #     psi *= VxExpGrid
        
    #     # if print_every_t > 0 and step % round(print_every_t / dt) == 0: 
    #     #     plotNow(t,psi)
    #     t += dt 

    # for step in range(steps):
    #     # cosGrid = np.cos(2*kx*xlin[:,np.newaxis] + 2*kz*zlin + doppd*(t-tauMid) + phase)
    #     # VxExpGrid = np.exp(-(1j/hb) * 0.5*dt * VBF(t,tauMid,tauPi,V0FArg) * cosGrid )
    #     # psi *= VxExpGrid
    #     # phi = toMomentum(psi,swnf)
    #     # phi *= expPGrid
    #     # psi = toPosition(phi,swnf)
    #     # psi *= VxExpGrid
    #     # t += dt 
    #     # loop()
    #     (t,psi) = loop(t,psi,tauPi,tauMid,V0FArg,doppd,phase,swnf)

        
    if progress_bar:
        for step in tqdm(range(steps)):
            # loop()
            (t,psi) = loop(t,psi,tauPi,tauMid,V0FArg,doppd,phase,swnf,kkx,kkz)
    else:
        for step in range(steps):
            # loop()
            (t,psi) = loop(t,psi,tauPi,tauMid,V0FArg,doppd,phase,swnf,kkx,kkz)
    
    # if final_plot:
    #     print("ALL DONE")
    #     plotNow(t,psi)
    return (t,psi,phi)


# In[44]:


_ = numericalEvolve(0, psi0np(1,1,0,0), 10*dt, final_plot=False, progress_bar=False)


# In[45]:


# test run 
# get_ipython().run_line_magic('timeit', 'numericalEvolve(0, psi0np(1,1,0,0), dt, final_plot=False, progress_bar=False)')
# M1 107 ms ± 2.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


# In[46]:


# @jit(forceobj=True, cache=True)
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


# In[47]:


_ = freeEvolve(0,psi0np(1,1,0,0),0.1,final_plot=False,logging=True)
# get_ipython().run_line_magic('timeit', 'freeEvolve(0,psi0np(1,1,0,0),0.1,final_plot=False,logging=False)')
#M1 85.6 ms ± 1.74 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


# In[48]:


(tBraggCenter,tBraggEnd,tBraggPi)


# In[49]:


# @jit(forceobj=True, cache=True)
def scanTauPiInnerEval(tPi, 
                       logging=True, progress_bar=True, 
                       ang=0, pmom=p, doppd=dopd, V0FArg=V0F, kkx=kx, kkz=kz):
    tauPi  = tPi
    tauMid = tauPi * 5
    tauEnd = tauPi * 10
    if logging:
        print("Testing parameters")
        print("tauPi =", round(tPi,6), "    \t tauMid =", round(tauMid,6), " \t tauEnd = ", round(tauEnd,6))
    # output = numericalEvolve(0, psi0np(2,2,pmom*np.cos(ang),pmom*np.sin(ang)), 
    # !!!! THIS !!!! 
    output = numericalEvolve(0, psi0ringNpOffset(5,1,pmom,0,5,0,pmom), 
                             tauEnd, tauPi, tauMid, doppd=doppd, 
                             final_plot=logging,progress_bar=progress_bar,
                             V0FArg=V0FArg,kkx=kkx,kkz=kkz)
#     if pbar != None: pbar.update(1)
    return output


# ## Scanning pulse duration

# In[124]:


dt*1000


# In[125]:


tBraggPi*1000


# In[ ]:





# In[127]:


# tPiTest = np.append(np.arange(0,0.05,-0.0005), 0) # note this is decending
    # tPiTest = np.arange(dt,3*dt,dt)
dddt = 0.001 * 0.05
tPiTest = np.flip(np.linspace(0,dddt*600,20)) ##AHHHH
l.info(f"#tPiTest = {len(tPiTest)}, max={tPiTest[0]*1000}, min={tPiTest[-1]*1000} us")

plt.figure(figsize=(12,5))
def plot_inner_helper():
    for (i, tauPi) in enumerate(tPiTest):
        if tauPi == 0: continue
        tauMid = tauPi * 5 
        tauEnd = tauPi * 10 
        tlinspace = np.arange(0,tauEnd,dt)
        plt.plot(tlinspace*1000, VBF(tlinspace, tauMid, tauPi),
                 linewidth=0.5,alpha=0.9
            )
plt.subplot(2,1,1)
plot_inner_helper()
plt.ylabel("$V(t)$")

plt.subplot(2,1,2)
plot_inner_helper()
plt.xlim([0,5])
plt.xlabel("$t \ (μs)$ ")
plt.ylabel("$V(t)$")


title="bragg_strength_V0"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show(block=False)


# In[53]:


# len(tPiTest)*0.1*tPiTest[0]*10/dt/3600


# In[54]:


# tPiTestRun = scanTauPiInnerEval(dt, False, True,0,p,0*dopd,0.0001*V0F)


# In[ ]:





# In[55]:


tPiTest


# In[128]:


print(tPiTest[-2])
tPiTestRun = scanTauPiInnerEval(tPiTest[-2], False, True,0,p,0*dopd,VR)


# In[176]:


testFreeEv1 = freeEvolve(0,psi0ringNpOffset(5,1,p,0,5,0,p),
                        tPiTest[0]*10,final_plot=False,logging=False)
testFreeEv2 = freeEvolve(0,psi0ringNpOffset(5,1,p,0,5,0,p),
                         tPiTest[0]*20,final_plot=False,logging=False)


# In[67]:


# tPiOutput = Parallel(n_jobs=N_JOBS)(
#     delayed(lambda i: (i, scanTauPiInnerEval(i, False, False,0,p,0*dopd,VR)[:2]) )(i)
#     for i in tqdm(tPiTest)
# )   #### THIS THING TAKE A FEW MIN (or Hours)


# In[177]:


# # psi = tPiOutput[-1][1][1]
# # psi = tPiTestRun[1]
# psi = testFreeEv1[1]
# plot_psi(psi)
# # (swnf, phi) = phiAndSWNF(psi)
# plot_mom(psi,5,5)


# In[143]:


# tPiOutputFramesDir = []
# os.makedirs(output_prefix+"tPiScan", exist_ok=True)
# for (ti, tPi) in enumerate(tPiTest):
#     print(f"Exporting frame {ti}, tPi={tPi*1000}us",end="\r")
#     psi = tPiOutput[ti][1][1]
#     plot_psi(psi, False)
#     plt.savefig(output_prefix+f"tPiScan/psi-({ti},{tPi}).png",dpi=600)
#     tPiOutputFramesDir.append(output_prefix+f"tPiScan/psi-({ti},{tPi}).png")
#     plt.close()
#     plot_mom(psi,5,5,False)
#     plt.savefig(output_prefix+f"tPiScan/phi-({ti},{tPi}).png",dpi=600)
#     plt.close()
# print("DONE \t\t\t ", end="\r")


# In[79]:


# dpz/p


# In[111]:


hbar_k_transfers = np.arange(-7,7+2,+2)
# pzlinIndexSet = np.zeros((len(hbar_k_transfers), len(pxlin)), dtype=bool)
pxlinIndexSet = np.zeros((len(hbar_k_transfers), len(pzlin)), dtype=bool)
cut_p_width = 2*dpz/p
for (j, hbar_k) in enumerate(hbar_k_transfers):
    # pzlinIndexSet[j] = abs(pxlin/(hb*k) - hbar_k) <= cut_p_width
    pxlinIndexSet[j] = abs(pzlin/p + hbar_k) <= cut_p_width
    # print(i,hbar_k)


# In[112]:


# np.sum(pxlinIndexSet,axis=1)


# In[113]:


# plt.figure(figsize=(4,4))
# plt.imshow(pxlinIndexSet.T,interpolation='none',aspect=1,extent=[-7,7,-pzmax/p,pzmax/p])
# # plt.axvline(x=1001, linewidth=1, alpha=0.7)
#
# plt.xlabel("index")
# plt.ylabel("pz")
# title="hbar_k_pxlin_integration_range"
# # plt.savefig("output/"+title+".pdf", dpi=600)
# # plt.savefig("output/"+title+".png", dpi=600)
# plt.show(block=False)


# In[114]:


# hbar_k_transfers


# In[115]:


# phiDensityGrid = np.zeros((len(tPiTest), pxlin.size))
phiDensityGrid = np.zeros((len(tPiTest), pzlin.size))
phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))

# for i in tqdm(range(len(tPiTest))):
#     item = tPiOutput[i]
#     (swnf, phi) = phiAndSWNF(item[1][1])
#     phiAbsSq = np.abs(phi)**2
#     # phiX = np.trapz(phiAbsSq, pzlin,axis=1)
#     phiZ = np.trapz(phiAbsSq, pzlin,axis=0)
#     # phiDensityGrid[i] = phiX
#     phiDensityGrid[i] = phiZ
#
#     for (j, hbar_k) in enumerate(hbar_k_transfers):
#         # index = pzlinIndexSet[j]
#         index = pxlinIndexSet[j]
#         # phiDensityGrid_hbark[i,j] = np.trapz(phiX[index], pxlin[index])
#         phiDensityGrid_hbark[i,j] = np.trapz(phiZ[index], pzlin[index])


# In[ ]:





# In[116]:


# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
#
# nxm = int((nx-1)/2)
# nx2 = int((nx-1)/2)
# plt.imshow(phiDensityGrid[:,nxm-nx2:nxm+nx2],
#            # extent=[pxlin[nxm-nx2]/(hb*k),pxlin[nxm+nx2]/(hb*k),0,len(tPiTest)],
#            extent=[pzlin[nxm-nx2]/(hb*k),pzlin[nxm+nx2]/(hb*k),0,len(tPiTest)],
#            interpolation='none',aspect=0.15)
# # plt.imshow(phiDensityGrid,
# #            extent=[-pxmax/(hb*k),pxmax/(hb*k),1,len(tPiTest)+1],
# #            interpolation='none',aspect=1)
# # ax = plt.gca()
# # for t in tPiTest:
# #     plt.axhline(y=t/dt,color='white',alpha=1,linewidth=0.05,linestyle='-')
# ax = plt.gca()
# ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
# plt.ylabel("$dt =$"+str(dt*1000) + "$\mu \mathrm{s}$")
# # plt.xlabel("$p_x \ (\hbar k)$")
# plt.xlabel("$p_z \ (\hbar k)$")
#
#
# plt.subplot(1,2,2)
# plt.imshow(phiDensityGrid_hbark,
#            extent=[hbar_k_transfers[0],hbar_k_transfers[-1],0,len(tPiTest)],
#            interpolation='none',aspect=0.15)
# # plt.xlabel("$p_x \ (\hbar k)$ integrated block")
# plt.xlabel("$p_z \ (\hbar k)$ integrated block")
#
# title="mom_dist_at_diff_angle"
# # plt.savefig("output/"+title+".pdf", dpi=600)
# # plt.savefig("output/"+title+".png", dpi=600)
# plt.show(block=False)


# In[117]:


# phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)


# In[118]:


# plt.figure(figsize=(11,5))
# for (i, hbar_k) in enumerate(hbar_k_transfers):
#     if abs(hbar_k) >3: continue
#     if   hbar_k > 0: style = '+-'
#     elif hbar_k < 0: style = 'x-'
#     else:            style = '.-'
#     plt.plot(tPiTest*1000, phiDensityGrid_hbark[:,i]/phiDensityNormFactor[i],
#              style, linewidth=1,alpha=0.5, markersize=5,
#              label=str(hbar_k)+"$\hbar k$",
#             )
#
# plt.legend(loc=1,ncols=2)
# plt.ylabel("$normalised \int |\phi(p)| dp$ around region ($\pm$"+str(cut_p_width)+")")
# plt.xlabel("$t_\pi \ (\mu s)$")
# # plt.axhline(y=np.cos(pi  )**2,color='gray',linewidth=1,alpha=0.5)  # 2*pi pulse
# # plt.axhline(y=np.cos(pi/2)**2,color='c',linewidth=1,alpha=0.5)     # pi   pulse
# # plt.axhline(y=np.cos(pi/4)**2,color='violet',linewidth=1,alpha=0.5) # pi/2 pulse
# # plt.axhline(y=np.cos(pi/8)**2,color='orange',linewidth=1,alpha=0.5)# pi/4 pulse
#
# # plt.axvline(x=tPiTest[81]*1000,color='c',linewidth=1,alpha=0.5)      # pi   pulse
# # plt.axvline(x=tPiTest[90]*1000,color='violet',linewidth=1,alpha=0.5) # pi/2 pulse
# # plt.axvline(x= 9*dt*1000,color='orange',linewidth=1,alpha=0.5)  # pi/4 pulse
#
# # plt.text((1+39)*dt*1000, 1, "$\pi$",color='c')
# # plt.text((1+21)*dt*1000, 1, "$\pi/2$",color='violet')
# # plt.text((1+ 9)*dt*1000, 1, "$\pi/4$",color='orange')
#
# title = "bragg_pulse_duration_test_labeled"
# # plt.savefig("output/"+title+".pdf", dpi=600)
# # plt.savefig("output/"+title+".png", dpi=600)
#
# plt.show(block=False)


# In[119]:


# hbar_k_transfers


# In[120]:


# np.argmax(phiDensityGrid_hbark[:,3]/phiDensityNormFactor[3])


# In[89]:


# tPiTest[6]*1000


# In[105]:


# phiDensityGrid_hbark[12,3]/phiDensityNormFactor[3]


# In[ ]:





# In[ ]:





# In[ ]:





# ## Setting up intensity duration scan

# In[53]:


dddt = 0.001 * 0.05
tPiTest = np.flip(np.linspace(0,dddt*400,100)) ##AHHHH
l.info(f"#tPiTest = {len(tPiTest)}, max={tPiTest[0]*1000}, min={tPiTest[-1]*1000} us")
l.info(f"tPiTest: {tPiTest}")

plt.figure(figsize=(12,5))
def plot_inner_helper():
    for (i, tauPi) in enumerate(tPiTest):
        if tauPi == 0: continue
        tauMid = tauPi * 5 
        tauEnd = tauPi * 10 
        tlinspace = np.arange(0,tauEnd,dt)
        plt.plot(tlinspace*1000, VBF(tlinspace, tauMid, tauPi),
                 linewidth=0.5,alpha=0.9
            )
plt.subplot(2,1,1)
plot_inner_helper()
plt.ylabel("$V(t)$")

plt.subplot(2,1,2)
plot_inner_helper()
plt.xlim([0,5])
plt.xlabel("$t \ (μs)$ ")
plt.ylabel("$V(t)$")


title="bragg_strength_V0"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show(block=False)


# In[54]:


omegaRabi


# In[123]:


intensityScan = np.linspace(0.1,2,50)
l.info(f"intensityScan: {intensityScan}")
omegaRabiScan = (linewidth*np.sqrt(intensityScan/intenSat/2))**2 /2/detuning
l.info(f"omegaRabiScan: {omegaRabiScan}")
VRScan = 2*hb*omegaRabiScan*0.001
l.info(f"VRScan: {VRScan}")
l.info(f"VRScan/VR: {VRScan/VR}")


# In[124]:


intensityWidthGrid = []
for (VRi,VRs) in enumerate(VRScan):
    tempTRow = []
    for (ti, tP) in enumerate(tPiTest):
        tempTRow.append((VRs,tP))
    intensityWidthGrid.append(tempTRow)


# In[125]:


VRs_grid, tP_grid = np.meshgrid(VRScan, tPiTest, indexing='ij')
intensityWidthGrid = np.stack((VRs_grid, tP_grid), axis=-1)


# In[126]:


intensityWidthGrid[:,1]


# In[127]:


intensityWidthGrid[0,:]


# In[128]:


ksz=kz
ksx=kx


# In[ ]:





# In[ ]:


VRScanOutput = []
for (VRi,VRs) in enumerate(VRScan):
    l.info(f"Computing VRi={VRi}, VRs={VRs}")
    tPiOutput = Parallel(n_jobs=N_JOBS)(
        delayed(lambda i: (i, scanTauPiInnerEval(i, False, False,0,p,0*dopd,VRs)[:2],ksz,ksx) )(i) 
        for i in tqdm(tPiTest)
    )   #### THIS THING TAKE A FEW MIN (or Hours)
    phiDensityGrid = np.zeros((len(tPiTest), pzlin.size))
    phiDensityGrid_hbark = np.zeros((len(tPiTest),len(hbar_k_transfers)))
    
    # tPiOutputFramesDir = []
    output_prefix_tPiVscan = output_prefix+f"tPiScan ({VRi},{VRs})"
    os.makedirs(output_prefix_tPiVscan, exist_ok=True)
    for (ti, tPi) in enumerate(tPiTest):
        print(f"Exporting frame {ti}, tPi={tPi*1000}us",end="\r")
        psi = tPiOutput[ti][1][1]
        plot_psi(psi, False)
        plt.savefig(output_prefix_tPiVscan+f"/psi-({ti},{tPi}).png",dpi=600)
        # tPiOutputFramesDir.append(output_prefix+f"tPiScan/psi-({ti},{tPi}).png")
        plt.close()
        plot_mom(psi,5,5,False)
        plt.savefig(output_prefix_tPiVscan+f"/phi-({ti},{tPi}).png",dpi=600)
        plt.close()
    print("DONE \t\t\t ", end="\r")
    
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
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    nxm = int((nx-1)/2)
    nx2 = int((nx-1)/2)
    plt.imshow(phiDensityGrid[:,nxm-nx2:nxm+nx2], 
               extent=[pzlin[nxm-nx2]/(hb*k),pzlin[nxm+nx2]/(hb*k),0,len(tPiTest)], 
               interpolation='none',aspect=0.15)
    ax = plt.gca()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.ylabel("$dt =$"+str(dt*1000) + "$\mu \mathrm{s}$")
    plt.xlabel("$p_z \ (\hbar k)$")
    plt.subplot(1,2,2)
    plt.imshow(phiDensityGrid_hbark, 
               extent=[hbar_k_transfers[0],hbar_k_transfers[-1],0,len(tPiTest)], 
               interpolation='none',aspect=0.15)
    plt.xlabel("$p_z \ (\hbar k)$ integrated block")
    
    title="mom_dist_at_diff_angle"
    plt.savefig(output_prefix_tPiVscan+"/"+title+".pdf", dpi=600)
    plt.savefig(output_prefix_tPiVscan+"/"+title+".png", dpi=600)
    # plt.show(block=False)
    plt.close()
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)
    plt.figure(figsize=(11,5))
    for (i, hbar_k) in enumerate(hbar_k_transfers):
        if abs(hbar_k) >3: continue
        if   hbar_k > 0: style = '+-'
        elif hbar_k < 0: style = 'x-'
        else:            style = '.-'
        plt.plot(tPiTest*1000, phiDensityGrid_hbark[:,i]/phiDensityNormFactor[i],
                 style, linewidth=1,alpha=0.5, markersize=5,
                 label=str(hbar_k)+"$\hbar k$",
                )
    
    plt.legend(loc=1,ncols=2)
    plt.ylabel("$normalised \int |\phi(p)| dp$ around region ($\pm$"+str(cut_p_width)+")")
    plt.xlabel("$t_\pi \ (\mu s)$")
    title = "bragg_pulse_duration_test_labeled"
    plt.savefig(output_prefix_tPiVscan+"/"+title+".pdf", dpi=600)
    plt.savefig(output_prefix_tPiVscan+"/"+title+".png", dpi=600)
    # plt.show(block=False)
    plt.close()
    
    indMax = np.argmax(phiDensityGrid_hbark[:,3]/phiDensityNormFactor[3])
    gc.collect()


# In[229]:


vtSliceM1 = np.empty((len(VRScan),len(tPiTest)))
for (VRi,VRs) in enumerate(VRScan):
    phiDensityGrid_hbark = VRScanOutput[VRi][2]
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)
    vtSliceM1[VRi] = phiDensityGrid_hbark[:,3]/phiDensityNormFactor


# In[241]:

plt.close()
plt.imshow(np.fliplr(np.flipud(vtSliceM1)),aspect=3, 
           extent=[tPiTest[-1]*1000,tPiTest[0]*1000,intensityScan[0],intensityScan[-1]],
           cmap=plt.get_cmap('viridis', 20)
          )
plt.colorbar(ticks=np.linspace(0,1,21))
plt.xlabel("Pulse width $\sigma$ $\mu s$")
plt.ylabel("Intensity $\mathrm{mW/mm^2}$")
title = "Transfer fraction of halo into -1$\hbar k$ state"
plt.title(title)
plt.savefig(output_prefix+"/"+title+".pdf", dpi=600)
plt.savefig(output_prefix+"/"+title+".png", dpi=600)
plt.show(block=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




