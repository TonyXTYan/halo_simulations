#!/usr/bin/env python
# coding: utf-8

# # One Particle

# In[1]:


get_ipython().system('python -V')


# In[2]:


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

from joblib import Parallel, delayed

from tqdm.notebook import tqdm
from datetime import datetime
import time

import pyfftw


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots


# In[3]:


N_JOBS=-1-3
nthreads=2


# In[4]:


import gc
gc.enable()


# In[5]:


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


# In[6]:


plt.set_loglevel("warning")


# In[7]:


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


# In[8]:


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


# In[9]:


l.info(f"""nthreads = {nthreads}
N_JOBS = {N_JOBS}""")


# In[10]:


# np.show_config()


# In[11]:


nx = 120+1
nz = 120+1
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


# In[12]:


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


# In[13]:


l.info(f"""rotate phase per dt for m3 = {1j*hb*dt/(2*m3*dx*dz)} \t #want this to be small
rotate phase per dt for m4 = {1j*hb*dt/(2*m4*dx*dz)} 
number of grid points = {round(nx*nz/1000/1000,3)} (million)
minutes per grid op = {round((nx*nz)*0.001*0.001/60, 3)} \t(for 1μs/element_op)
""")


# In[14]:


wavelength = 1.083 #Micrometers
beam_angle = 90
k = sin(beam_angle*pi/180/2) * 2*pi / wavelength # effective wavelength
klab = k

#alternatively 
# k = pi / (4*dx)
k = pi / (6*dx)
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


# In[15]:


xmax**2 * m3 * 6 / (hb*pi*(nx-1))


# In[16]:


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


# In[17]:


hb*pi*(nx-1) / (2*m3*xmax*6)


# In[18]:


l.info(f"""xmax/v3 = {xmax/v3} ms is the time to reach boundary
zmax/v3 = {zmax/v3}
""")


# In[ ]:





# In[19]:


#### WARNING:
###  These frequencies are in Hz, 
#### This simulation uses time in ms, 1Hz = 0.001 /ms
a4 = 0.007512 # scattering length µm
intensity1 = 0.1 # mW/mm^2 of beam 1
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


# In[20]:


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


# In[ ]:





# In[21]:


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


# In[22]:


l.info(f"""hb*k**2/(2*m3) = {hb*k**2/(2*m3)} \t/ms
hb*k**2/(2*m4) = {hb*k**2/(2*m4)}
(hb*k**2/(2*m3))**-1 = {(hb*k**2/(2*m3))**-1} \tms
(hb*k**2/(2*m4))**-1 = {(hb*k**2/(2*m4))**-1}
2*pi*hb*k**2/(2*m3) = {2*pi*hb*k**2/(2*m3)} \t rad/ms
2*pi*hb*k**2/(2*m4) = {2*pi*hb*k**2/(2*m4)}
omegaRabi = {omegaRabi*0.001} \t/ms
tBraggPi = {tBraggPi} ms
""")


# In[23]:


def V(t):
    return V0 * (2*pi)**-0.5 * tBraggPi**-1 * np.exp(-0.5*(t-tBraggCenter)**2 * tBraggPi**-2)

def VB(t, tauMid, tauPi):
    return V0 * (2*pi)**-0.5 * tauPi**-1 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)

V0F = 50*1000
def VBF(t, tauMid, tauPi, V0FArg=V0F):
    return V0FArg * (2*pi)**-0.5 * np.exp(-0.5*(t-tauMid)**2 * tauPi**-2)


# In[24]:


l.info(f"term infront of Bragg potential {1j*(dt/hb)}")
l.info(f"max(V) {1j*(dt/hb)*V(tBraggCenter)}")


# In[25]:


def VS(ttt, mid, wid, V0=VR):
    return V0 * 0.5 * (1 + np.cos(2*np.pi/wid*(ttt-mid))) * \
            (-0.5*wid+mid<ttt) * (ttt<0.5*wid+mid)


# In[26]:


tbtest = np.arange(tBraggCenter-5*tBraggPi,tBraggCenter+5*tBraggPi,dt)
plt.plot(tbtest, VBF(tbtest,tBraggPi*5,tBraggPi))
plt.plot(tbtest, VS(tbtest,tBraggPi,tBraggPi*2,0.3*V0F))
plt.show()
l.info(f"max(V) {1j*(dt/hb)*VBF(tBraggCenter,tBraggPi*5,tBraggPi)}")


# In[27]:


V(tBraggCenter)


# In[28]:


VBF(tBraggCenter,tBraggPi*5,tBraggPi)


# In[29]:


np.trapz(V(tbtest),tbtest) # this should be V0


# In[30]:


xlin = np.linspace(-xmax,+xmax, nx)
zlin = np.linspace(-zmax,+zmax, nz)
psi=np.zeros((nx,nz),dtype=complex)
zones = np.ones(nz)
xgrid = np.tensordot(xlin,zones,axes=0)
# cosGrid = np.cos(2*k*xgrid)
cosGrid = np.cos(2 * kx * xlin[:, np.newaxis] + 2 * kz * zlin)


# In[31]:


if abs(dx - (xlin[1]-xlin[0])) > 0.0001: l.error("AHHHHx")
if abs(dz - (zlin[1]-zlin[0])) > 0.0001: l.error("AHHHHz")


# In[32]:


l.info(f"{round(psi.nbytes/1000/1000 ,3)} MB of data used to store psi")


# In[33]:


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
plt.plot(cosGrid[0,:ncrop],alpha=0.9,linewidth=0.5)

title="bragg_potential_grid"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show()


# In[ ]:





# In[34]:


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


# In[35]:


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
        


# In[36]:


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


# In[37]:


expPGrid = np.zeros((nx,nz),dtype=complex)
for indx in range(nx):
    expPGrid[indx, :] = np.exp(-(1j/hb) * (0.5/m3) * (dt) * (pxlin[indx]**2 + pzlin**2))  


# In[38]:


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


# In[39]:


def psi0ringUnNormOffset(x,z,pr=p,mur=10,sg=sg,xo=0,zo=0,pxo=0,pzo=0):
    return 1 \
            * np.exp(-0.5*( mur - np.sqrt((x-xo)**2 + (z-zo)**2) )**2 / sg**2) \
            * np.exp(+(1j/hb) * (((x-xo)**2 + (z-zo)**2)**0.5 * pr + x*pxo+z*pzo))
def psi0ringNpOffset(mur=1,sg=1,pr=p,xo=0,zo=0,pxo=0,pzo=0):
    psi = np.zeros((nx,nz),dtype=np.complex128)
    for ix in range(1,nx-1):
        x = xlin[ix]
        psi[ix][1:-1] = psi0ringUnNormOffset(x,zlin[1:-1],pr,mur,sg,xo,zo,pxo,pzo)
    norm = np.sum(np.abs(psi)**2)*dx*dz
    psi *= 1/sqrt(norm)
    return psi


# In[ ]:





# In[ ]:





# In[40]:


# psi = psi0np(5,5,0,0)
# psi = psi0np(5,5,-0.5*p,0)
# psi = psi0np(1,1,p,p)
# psi = psi0np(1,1,0,0)

def phiAndSWNF(psi):
    phiUN = np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))
    # superWeirdNormalisationFactorSq = np.trapz(np.trapz(np.abs(phiUN)**2, pxlin, axis=0), pzlin)
    superWeirdNormalisationFactorSq = np.sum(np.abs(phiUN)**2)*dpx*dpz
    swnf = sqrt(superWeirdNormalisationFactorSq)
    phi = phiUN/swnf
    return (swnf, phi)
    
# psi = psi0ringNp(4,2,p)
psi = psi0ringNpOffset(5,5,p,0,5,0,p)
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
plt.show()

plot_mom(psi,4,4,False)
title="init_ring_phi"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()


# In[ ]:





# In[41]:


def toMomentum(psi, swnf):
    return np.fliplr(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(psi,threads=nthreads,norm='ortho')))/swnf
def toPosition(phi, swnf):
    return pyfftw.interfaces.numpy_fft.ifft2(np.fft.ifftshift(np.fliplr(phi*swnf)),threads=nthreads,norm='ortho')


# In[42]:


def plotNow(t, psi):
        print("time =", round(t*1000,4), "µs")
        print(np.sum(np.abs(psi)**2)*dx*dz,"|psi|^2")
        print(np.sum(np.abs(phi)**2)*dpx*dpz,"|phi|^2")
        plot_psi(psi)
        plot_mom(psi)

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


# In[43]:


_ = numericalEvolve(0, psi0np(1,1,0,0), dt, final_plot=False, progress_bar=False)


# In[44]:


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


# In[45]:


_ = freeEvolve(0,psi0np(1,1,0,0),0.1,final_plot=False,logging=True)


# In[46]:


(tBraggCenter,tBraggEnd,tBraggPi)


# In[47]:


# short test run
_ = numericalEvolve(0, psi0np(3,3,0.5*p,0), 2*dt,progress_bar=False,final_plot=False)


# In[48]:


def scanTauPiInnerEval(tPi, 
                       logging=True, progress_bar=True, 
                       ang=0, pmom=p, doppd=dopd, V0FArg=V0F, kkx=kx, kkz=kz):
    tauPi  = tPi
    tauMid = tauPi / 2 
    tauEnd = tauPi 
    if logging:
        print("Testing parameters")
        print("tauPi =", round(tPi,6), "    \t tauMid =", round(tauMid,6), " \t tauEnd = ", round(tauEnd,6))
    # output = numericalEvolve(0, psi0np(2,2,pmom*np.cos(ang),pmom*np.sin(ang)), 
    output = numericalEvolve(0, psi0ringNpOffset(5,5,pmom,0,5,0,pmom), 
                             tauEnd, tauPi, tauMid, doppd=doppd, 
                             final_plot=logging,progress_bar=progress_bar,
                             V0FArg=V0FArg,kkx=kkx,kkz=kkz
                            )
#     if pbar != None: pbar.update(1)
    return output


# In[98]:


tPDelta = 40*dt
tPiTest = np.append(np.arange(0.8,0,-tPDelta), 0) # note this is decending
    # tPiTest = np.arange(dt,3*dt,dt)
l.info(f"#tPiTest = {len(tPiTest)}, max={tPiTest[0]*1000}, min={tPiTest[-1]*1000} us")
l.info(f"tPiTest: {tPiTest}")

plt.figure(figsize=(12,5))
def plot_inner_helper():
    for (i, tauPi) in enumerate(tPiTest):
        if tauPi == 0: continue
        tauMid = tauPi / 2 
        tauEnd = tauPi * 1
        tlinspace = np.arange(0,tauEnd,dt)
        # plt.plot(tlinspace, VBF(tlinspace, tauMid, tauPi),
        #          linewidth=0.5,alpha=0.9
        #     )
        plt.plot(tlinspace, VS(tlinspace, tauMid, tauPi),
                 linewidth=0.5,alpha=0.9
            )
plt.subplot(2,1,1)
plot_inner_helper()
plt.ylabel("$V(t)$")

plt.subplot(2,1,2)
plot_inner_helper()
plt.xlim([0,0.02])
plt.xlabel("$t \ (ms)$ ")
plt.ylabel("$V(t)$")


title="bragg_strength_V0"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)

plt.show()


# In[ ]:





# In[50]:


tPiScanTimeStart = datetime.now()
tPiOutput = Parallel(n_jobs=N_JOBS)(
    delayed(lambda i: (i, scanTauPiInnerEval(i, False, False,0,p,0*dopd,VR)[:2]) )(i) 
    for i in tqdm(tPiTest)
) 
tPiScanTimeEnd = datetime.now()
tPiScanTimeDelta = tPiScanTimeEnd-tPiScanTimeStart
l.info(f"""Time to run one scan: {tPiScanTimeDelta}""")


# In[51]:


tPiOutput[30][1][0]


# In[52]:


tPiTest[30]


# In[53]:


# psi = tPiOutput[-30][1][1]
psi = tPiOutput[-5][1][1]
# psi = tPiTestRun[1]
# psi = testFreeEv1[1]
plot_psi(psi)
# (swnf, phi) = phiAndSWNF(psi)
plot_mom(psi,5,5)


# In[ ]:





# In[54]:


hbar_k_transfers = np.arange(-5,5+1,+2)
# pzlinIndexSet = np.zeros((len(hbar_k_transfers), len(pxlin)), dtype=bool)
pxlinIndexSet = np.zeros((len(hbar_k_transfers), len(pzlin)), dtype=bool)
cut_p_width = 1.5*dpz/p
for (j, hbar_k) in enumerate(hbar_k_transfers):
    # pzlinIndexSet[j] = abs(pxlin/(hb*k) - hbar_k) <= cut_p_width
    pxlinIndexSet[j] = abs(pzlin/p + hbar_k) <= cut_p_width
    # print(i,hbar_k)


# In[55]:


hbar_k_transfers


# In[56]:


np.sum(pxlinIndexSet,axis=1)


# In[57]:


plt.figure(figsize=(4,4))
plt.imshow(pxlinIndexSet.T,interpolation='none',aspect=0.5,extent=[-2,2,-pzmax/p,pzmax/p])
# plt.imshow(pzlinIndexSet,interpolation='none',aspect=5)
# plt.axvline(x=1001, linewidth=1, alpha=0.7)

title="hbar_k_pxlin_integration_range"
# plt.savefig("output/"+title+".pdf", dpi=600)
# plt.savefig("output/"+title+".png", dpi=600)
plt.show()


# In[58]:


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


# In[59]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

nxm = int((nx-1)/2)
nx2 = int((nx-1)/2)
mom_dist_at_diff_angle_phi_asss=0.04
mom_dist_at_diff_angle_den_asss=0.04
plt.imshow(phiDensityGrid[:,nxm-nx2:nxm+nx2], 
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


# In[60]:


# phiDensityNormFactor = np.sum(phiDensityGrid_hbark,axis=1)
phiDensityNormFactor = np.trapz(phiDensityGrid_hbark,axis=1)
# phiDensityNormed = np.zeros(phiDensityGrid_hbark.shape)
# for i in range(len(hbar_k_transfers)):
#     phiDensityNormed[:,i] = phiDensityGrid_hbark[:,i]/phiDensityNormFactor[i]
phiDensityNormed = phiDensityGrid_hbark / phiDensityNormFactor[:, np.newaxis]


# In[61]:


# phiDensityNormFactor


# In[62]:


os.makedirs(output_prefix+"tPiScan", exist_ok=True)


# In[63]:


plt.figure(figsize=(11,5))
for (i, hbar_k) in enumerate(hbar_k_transfers):
    if abs(hbar_k) >5: continue
    if   hbar_k > 0: style = '+-'
    elif hbar_k < 0: style = 'x-'
    else:            style = '.-'
    plt.plot(tPiTest*1000, phiDensityNormed[:,i],
             style, linewidth=1,alpha=0.5, markersize=5,
             label=str(hbar_k)+"$\hbar k$",
            )

plt.legend(loc=1,ncols=3)
plt.ylabel("$normalised \int |\phi(p)| dp$ around region ($\pm$"+str(cut_p_width)+")")
plt.xlabel("$t_\pi \ (\mu s)$")
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

title = "bragg_pulse_duration_test_labeled"
plt.savefig(output_prefix+"/tPiScan/"+title+".pdf", dpi=600)
plt.savefig(output_prefix+"/tPiScan/"+title+".png", dpi=600)

plt.show()


# In[64]:


hbarkInd = 2  # Index of target state
hbarkInI = 3  # Index of original state 


# In[65]:


indPDNPi = np.argmax(phiDensityNormed[:,hbarkInd])
l.info(f"""max transfer (π) to -1hbk at σt {tPiTest[indPDNPi]*1000} μs 
with efficiency to -1hbk: {phiDensityNormed[indPDNPi,hbarkInd]}""")


# In[66]:


pi2searchHelper=np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5)
indPDNPM = np.argmin(pi2searchHelper)
l.info(f"""max mirror (π/2) between ±1hbk at σt {tPiTest[indPDNPM]*1000} μs 
with transfer fraction to -1hbk of {phiDensityNormed[indPDNPM,hbarkInd]}
with transfer fraction to +1hbk of {phiDensityNormed[indPDNPM,hbarkInI]}""")


# In[67]:


plt.plot(tPiTest*1000,pi2searchHelper,'.-',alpha=0.7,label='helper')
plt.plot(tPiTest*1000,np.abs(phiDensityNormed[:,hbarkInd]-0.3),'.-',alpha=0.5,label="-1")
plt.plot(tPiTest*1000,np.abs(phiDensityNormed[:,hbarkInI]-0.3),'.-',alpha=0.5,label="+1")
plt.legend(ncols=3)
plt.ylabel("fraction")
plt.xlabel("$t_\pi \ (\mu s)$")
plt.show()


# In[68]:


l.info(f"""indPDNPi = {indPDNPi} \ttPiTest[indPDNPi] = {round(tPiTest[indPDNPi]*1000,2)} μs \teff {round(phiDensityNormed[indPDNPi,hbarkInd],4)}
indPDNPM = {indPDNPM} \ttPiTest[indPDNPM] = {round(tPiTest[indPDNPM]*1000,2)} μs \tef- {round(phiDensityNormed[indPDNPM,hbarkInd],4)} \tef+ {round(phiDensityNormed[indPDNPM,hbarkInI],4)}""")


# In[69]:


(indPDNPi, tPiTest[indPDNPi], phiDensityNormed[indPDNPi,hbarkInd])


# In[70]:


(indPDNPM, tPiTest[indPDNPM], phiDensityNormed[indPDNPM,hbarkInd], phiDensityNormed[indPDNPM,hbarkInI])


# In[ ]:





# In[ ]:





# In[71]:


tPiScanOutputTimeStart = datetime.now()
os.makedirs(output_prefix+"tPiScan", exist_ok=True)
def tPiTestFrameExportHelper(ti, tPi, output_prefix_tPiVscan): 
    psi = tPiOutput[ti][1][1]
    plot_psi(psi, False)
    plt.savefig(output_prefix_tPiVscan+f"/psi-({ti},{tPi}).png",dpi=600)
    # tPiOutputFramesDir.append(output_prefix+f"tPiScan/psi-({ti},{tPi}).png")
    plt.close()
    plot_mom(psi,5,5,False)
    plt.savefig(output_prefix_tPiVscan+f"/phi-({ti},{tPi}).png",dpi=600)
    plt.close()
    plt.cla() 
    plt.clf() 
    plt.close('all')
    # plt.ioff() # idk, one of these should clear memroy issues?
    time.sleep(0.05)
    gc.collect()
    return(output_prefix_tPiVscan+f"/psi-({ti},{tPi}).png", 
           output_prefix_tPiVscan+f"/phi-({ti},{tPi}).png")

tPiOutputFramesDir = Parallel(n_jobs=-3, timeout=1000)(
    delayed(tPiTestFrameExportHelper)(ti, tPiTest[ti], output_prefix+"tPiScan")
    for ti in tqdm(range(len(tPiTest)))
)
tPiScanOutputTimeEnd = datetime.now()
tPiScanOutputTimeDelta = tPiScanOutputTimeEnd-tPiScanOutputTimeStart
l.info(f"""Time to output one scan: {tPiScanOutputTimeDelta}""")


# In[ ]:





# In[ ]:





# In[ ]:





# ## Intensity Scan

# In[72]:


isDelta = 0.003
intensityScan = np.arange(0.01,0.30+isDelta,isDelta)
l.info(f"""len(intensityScan): {len(intensityScan)}
intensityScan: {intensityScan}""")
omegaRabiScan = (linewidth*np.sqrt(intensityScan/intenSat/2))**2 /2/detuning
l.info(f"omegaRabiScan: {omegaRabiScan}")
VRScan = 2*hb*omegaRabiScan*0.001
l.info(f"VRScan: {VRScan}")
l.info(f"VRScan/VR: {VRScan/VR}")


# In[73]:


l.info(f"""len(intensityScan) = {len(intensityScan)}
Each scan takes time roughtly {tPiScanTimeDelta.seconds}s + {tPiScanOutputTimeDelta.seconds}s  
Estimate total scan time: {(tPiScanTimeDelta+tPiScanOutputTimeDelta)*len(intensityScan)}""")


# In[74]:


intensityScanParamNotes = []
for i in range(len(intensityScan)):
    intensityScanParamNotes.append((i, intensityScan[i], omegaRabiScan[i], VRScan[i]))


# In[75]:


l.info(f"""intensityScanParamNotes
(i, intensityScan[i], omegaRabiScan[i], VRScan[i]): 
{intensityScanParamNotes}""".replace("), (", "),\n("))


# In[76]:


intensityWidthGrid = []
for (VRi,VRs) in enumerate(VRScan):
    tempTRow = []
    for (ti, tP) in enumerate(tPiTest):
        tempTRow.append((VRs,tP))
    intensityWidthGrid.append(tempTRow)
VRs_grid, tP_grid = np.meshgrid(VRScan, tPiTest, indexing='ij')
intensityWidthGrid = np.stack((VRs_grid, tP_grid), axis=-1)


# In[77]:


intensityWidthGrid[:,0] # VRScan , tPiTest[0] 


# In[78]:


intensityWidthGrid[0,:] # VRScan[0], tPiTest


# In[ ]:





# In[ ]:





# In[79]:


ksz=kz
ksx=kx
VRScanOutput = []
VRScanOutputPi = [] 
VRScanOutputPM = []
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
    _ = Parallel(n_jobs=-3, timeout=1000)(
    delayed(tPiTestFrameExportHelper)(ti, tPiTest[ti], output_prefix_tPiVscan)
        for ti in tqdm(range(len(tPiTest)))
    )

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
    indPDNPi = np.argmax(phiDensityNormed[:,hbarkInd])
    indPDNPM = np.argmin(np.abs(phiDensityNormed[:,hbarkInd]-0.5)+np.abs(phiDensityNormed[:,hbarkInI]-0.5))
    VRScanOutputPi.append((indPDNPi, tPiTest[indPDNPi], phiDensityNormed[indPDNPi,hbarkInd]))
    VRScanOutputPM.append((indPDNPM, tPiTest[indPDNPM], phiDensityNormed[indPDNPM,hbarkInd], phiDensityNormed[indPDNPM,hbarkInI]))
    l.info(f"""indPDNPi = {indPDNPi} \ttPiTest[indPDNPi] = {round(tPiTest[indPDNPi]*1000,2)} μs \teff {round(phiDensityNormed[indPDNPi,hbarkInd],4)}
indPDNPM = {indPDNPM} \ttPiTest[indPDNPM] = {round(tPiTest[indPDNPM]*1000,2)} μs \tef- {round(phiDensityNormed[indPDNPM,hbarkInd],4)} \tef+ {round(phiDensityNormed[indPDNPM,hbarkInI],4)}""")
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    nxm = int((nx-1)/2)
    nx2 = int((nx-1)/2)
    plt.imshow(phiDensityGrid[:,nxm-nx2:nxm+nx2], 
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
        if abs(hbar_k) >3: continue
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


# In[80]:


# hbarkInd = 2
vtSliceM1 = np.empty((len(VRScan),len(tPiTest)))
for (VRi,VRs) in enumerate(VRScan):
    if VRi >= len(VRScanOutput): break
    phiDensityGrid_hbark = VRScanOutput[VRi][2]
    phiDensityNormFactor = np.trapz(phiDensityGrid_hbark)
    vtSliceM1[VRi] = phiDensityGrid_hbark[:,hbarkInd]/phiDensityNormFactor
print(hbar_k_transfers[hbarkInd])


# In[81]:


vtSliceM1.shape


# In[82]:


VRSOPi = np.array(VRScanOutputPi)
VRSOPM = np.array(VRScanOutputPM)


# In[100]:


cmapS = plt.get_cmap('viridis', 20).copy()
cmapS.set_bad(color='black')
plt.close()
plt.figure(figsize=(12,8))
plt.imshow(np.fliplr(np.flipud(vtSliceM1)),aspect=0.8*(tPiTest[0]-tPiTest[-1])/(intensityScan[-1]-intensityScan[0])*1000, 
           extent=[tPiTest[-1]*1000,(tPiTest[0]+tPDelta)*1000,intensityScan[0],intensityScan[-1]+isDelta],
           cmap=cmapS
          )
plt.colorbar(ticks=np.linspace(0,1,21))
plt.scatter(VRSOPi[:,1]*1000,intensityScan+0.5*isDelta,color='red',marker='.',s=8)
plt.scatter(VRSOPM[:,1]*1000,intensityScan+0.5*isDelta,color='fuchsia',marker='.',s=10)
plt.xlabel("Pulse width $\sigma$ $\mu s$")
plt.ylabel("Intensity $\mathrm{mW/mm^2}$")
title = f"Transfer fraction of halo into {hbar_k_transfers[hbarkInd]}$\hbar k$ state"
plt.title(title)
# plt.savefig(output_prefix+"/"+title+".pdf", dpi=600)
# plt.savefig(output_prefix+"/"+title+".png", dpi=600)
plt.show()


# In[131]:


rowList = []
for (i,v) in enumerate(VRSOPi):
    u = VRSOPM[i]
    # print(f"{i}, {round(intensityScan[i],4)}, {round(VRScan[i],1)} \t"
    #       +f"{v[0]}, {round(v[1],3)}, {round(v[2],4)} \t"
    #       +f"{u[0]}, {round(u[1],3)}, {round(u[2],4)}, {round(u[3],4)}"
    #  )
    rowList.append((i,intensityScan[i],VRScan[i],v[0],v[1],v[2],u[0],u[1],u[2],u[3]))
pulseOutput = pd.DataFrame(rowList, columns=["i","I","V","Pi","ts","ef","PM","ts","e-","e+"])


# In[136]:


pulseOutput


# In[134]:


pulseOutput.to_csv(output_prefix+"VRScanPulseDuration.csv")


# In[119]:


with pgzip.open(output_prefix+'/VRScanOutput'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(VRScanOutput, file) 

with pgzip.open(output_prefix+'/intensityScan'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(intensityScan, file) 

with pgzip.open(output_prefix+'/intensityWidthGrid'+output_ext, 'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(intensityWidthGrid, file) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




