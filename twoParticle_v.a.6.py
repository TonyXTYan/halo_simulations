# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: py311_he34sim
#     language: python
#     name: python3
# ---

# # Two Particles

# # Setting things up

# ### Loading packages (🔄)

# !python -V

import sys
print(sys.path)

# +
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

from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from colorsys import hls_to_rgb # for complex plot color

import pickle
import pgzip
import os
import platform
import logging
import sys
import psutil
import glob
import re

# os.environ["IMAGEIO_FFMPEG_EXE"]="/opt/miniconda3/bin/ffmpeg"
# os.environ["IMAGEIO_FFMPEG_EXE"]="/envs/py311_he34sim/bin/ffmpeg"
from moviepy.editor import ImageSequenceClip
import cv2

from IPython.display import display, clear_output, HTML

from joblib import Parallel, delayed

from tqdm.notebook import tqdm
# from tqdm import tqdm
from datetime import datetime
import time

import pyfftw

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots
# -

warnings.filterwarnings("ignore", category=DeprecationWarning)

# +
from numba import njit, jit, prange, objmode, vectorize
import numba
numba.set_num_threads(8)
from numba_progress import ProgressBar
N_JOBS=2#-3-1
nthreads=2

import gc
gc.enable()

# +
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

plt.set_loglevel("warning")
# -

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

l.info(f"This file is {output_prefix}logs.log")

# +
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
# print_ram_usage(locals().items(),10)
# print_ram_usage(globals().items(),10)



# ### Simulation Parameters (🔄)

dtypec = np.cdouble
dtyper = np.float64
l.info(f"dtypec = {dtypec}, dtyper = {dtyper}")

# +
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
# -

l.info(f"""rotate phase per dt for m3 = {1j*hb*dt/(2*m3*dx*dz)} \t #want this to be small
rotate phase per dt for m4 = {1j*hb*dt/(2*m4*dx*dz)} 
number of grid points = {round(nx*nz/1000/1000,3)} (million)
minutes per grid op = {round((nx*nz)*0.001*0.001/60, 3)} \t(for 1μs/element_op)
""")

# +
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
# -

v4scat = m3/(m3+m4)*v4
v3scat = m4/(m3+m4)*v4
l.info(f"""v3scat = {v3scat}
v4scat = {v4scat}""")

xmax/v3scat

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

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}
l.info(f"""hb*k**2/(2*m3) = {hb*k**2/(2*m3)} \t/ms
hb*k**2/(2*m4) = {hb*k**2/(2*m4)}
(hb*k**2/(2*m3))**-1 = {(hb*k**2/(2*m3))**-1} \tms
(hb*k**2/(2*m4))**-1 = {(hb*k**2/(2*m4))**-1}
2*pi*hb*k**2/(2*m3) = {2*pi*hb*k**2/(2*m3)} \t rad/ms
2*pi*hb*k**2/(2*m4) = {2*pi*hb*k**2/(2*m4)}
omegaRabi = {omegaRabi*0.001} \t/ms
tBraggPi = {tBraggPi} ms
""")


# +
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


# -

l.warning(f"""2D grid RAM: {(nx*nz)*(np.cdouble(1).nbytes)/1000/1000} MB
4D grid RAM: {(nx*nz)**2*(np.cdouble(1).nbytes)/1000/1000} MB""")

psi=np.zeros((nx,nz, nx,nz),dtype=np.cdouble)
l.info(f"psi RAM usage: {round(psi.nbytes/1000/1000 ,3)} MB")

process_py = psutil.Process(os.getpid())
def ram_py(): return process_py.memory_info().rss;
def ram_py_MB(): return (ram_py()/1000**2)
def ram_py_GB(): return (ram_py()/1000**3)
def ram_py_log(): l.info(str(round(ram_py()/1000**2,3)) + "MB of system memory used")
ram_sys_MB = psutil.virtual_memory().total/1e6
ram_sys_GB = psutil.virtual_memory().total/1e9

l.info(f"Current RAM usage: {round(ram_py_MB(),3)} MB")

# xgrid = np.tensordot(xlin, np.ones(nz, dtype=dtyper), axes=0)
# cosXGrid = np.cos(2*k*xgrid)
zgrid = np.tensordot(np.ones(nz),zlin, axes=0)
cosZGrid = np.cos(2*k*zgrid)
l.info(f"cosZGrid size {round(cosZGrid.nbytes/1000**2,3)} MB")

# +
# Parameters from scan 20240506-184909-TFF
V3pi1 = 0.08*VR # ratio corrent when using intensity=1
t3pi1 = 49.8e-3 # ms 
e3pi1 = 0.9950  # expected efficiency from scan
V3pi2 = 0.04*VR
t3pi2 = 49.9e-3
e3pi2 = (0.4994, 0.4990)
# TODO
# Below form gussing 20240521-013024-TFF and Rabi oscillation relations
V3pi4  = 0.02*VR
t3pi4  = 49.9e-3
e3pi4  = -1
V3pi34 = 0.06*VR
t3pi34 = 49.9e-3
e3pi34 = -1
V3pi54 = 0.10*VR
t3pi54 = 49.8e-3
e3pi54 = -1
V3pi32 = 0.12*VR
t3pi32 = 49.8e-3
e3pi32 = -1
V3pi74 = 0.14*VR
t3pi74 = 49.8e-3
e3pi74 = -1
V3pi21 = 0.16*VR
t3pi21 = 49.8e-3
e3pi21 = -1
# 20240507-212137-TFF
V4pi1 = 0.06*VR # ratio corrent when using intensity=1
t4pi1 = 66.4e-3 # ms 
e4pi1 = 0.9950  # expected efficiency from scan
V4pi2 = 0.03*VR
t4pi2 = 66.5e-3
e4pi2 = (0.4990, 0.4994)
# TODO
# Below form gussing 20240521-013024-TFF and Rabi oscillation relations
V4pi4 = 0.015*VR
t4pi4 = 66.5e-3
e4pi4 = -1
V4pi34 = 0.045*VR
t4pi34 = 66.5e-3
e4pi34 = -1
V4pi54 = 0.075*VR
t4pi54 = 66.4e-3
e4pi54 = -1
V4pi32 = 0.09*VR
t4pi32 = 66.4e-3
e4pi32 = -1
V4pi74 = 0.105*VR
t4pi74 = 66.4e-3
e4pi74 = -1
V4pi21 = 0.12*VR
t4pi21 = 66.4e-3
e4pi21 = -1
# 20240509-181745-TFF 
V3sc = 0.135*VR
t3sc = 22.8e-3 
e3sc = 0.4238 
# 20240511-222534-TFF
V4sc = 0.102*VR
t4sc = 30.1e-3 
e4sc = 0.4239 

tLongestPulse = max([t3pi1,t3pi2,t4pi1,t4pi2,t4sc])
tLongestMRPulse = max([t3pi1,t4pi1])
tLongestBSPulse = max([t3pi2,t4pi2,t3pi4,t4pi4])
NLongestBSPulse = round(tLongestBSPulse/dt)
# Interferometer sequence timings
# 20240512-005555-TFF
T_a34off = 150e-3 #ms time to turn of scattering potential (can run free particle propagator)
T_MR = 600e-3 #ms time of centre of mirror pulse
T_BS = T_MR*2 - t4sc + 0e-3 #ms TODO: find?
T_END = T_BS + tLongestPulse/2

T_MR_L = T_MR - 0.5*tLongestMRPulse
T_MR_R = T_MR + 0.5*tLongestMRPulse

T_BS_L = T_BS - 0.5*tLongestBSPulse
T_BS_R = T_BS + 0.5*tLongestBSPulse

T_FREE_DELTA_1 = round(T_MR_L - T_a34off,5)
T_FREE_DELTA_2 = round(T_BS_L - T_MR_R,5)
N_FREE_STEPS_1 = round(T_FREE_DELTA_1 / dt)
N_FREE_STEPS_2 = round(T_FREE_DELTA_2 / dt)

l.info(f"""(V3pi4, t3pi4, e3pi4) = \t{(V3pi4, t3pi4, e3pi4)}
(V3pi2, t3pi2, e3pi2) = \t{(V3pi2, t3pi2, e3pi2)}
(V3pi34, t3pi34, e3pi34) = \t{(V3pi34, t3pi34, e3pi34)}
(V3pi1, t3pi1, e3pi1) = \t{(V3pi1, t3pi1, e3pi1)}
(V3pi54, t3pi54, e3pi54) = \t{(V3pi54, t3pi54, e3pi54)}
(V3pi32, t3pi32, e3pi32) = \t{(V3pi32, t3pi32, e3pi32)}
(V3pi74, t3pi74, e3pi74) = \t{(V3pi74, t3pi74, e3pi74)}
(V3pi21, t3pi21, e3pi21) = \t{(V3pi21, t3pi21, e3pi21)}
(V3sc, t3sc, e3sc) = \t{(V3sc, t3sc, e3sc)}

(V4pi4, t4pi4, e4pi4) = \t{(V4pi4, t4pi4, e4pi4)}
(V4pi2, t4pi2, e4pi2) = \t{(V4pi2, t4pi2, e4pi2)}
(V4pi34, t4pi34, e4pi34) = \t{(V4pi34, t4pi34, e4pi34)}
(V4pi1, t4pi1, e4pi1) = \t{(V4pi1, t4pi1, e4pi1)}
(V4pi54, t4pi54, e4pi54) = \t{(V4pi54, t4pi54, e4pi54)}
(V4pi32, t4pi32, e4pi32) = \t{(V4pi32, t4pi32, e4pi32)}
(V4pi74, t4pi74, e4pi74) = \t{(V4pi74, t4pi74, e4pi74)}
(V4pi21, t4pi21, e4pi21) = \t{(V4pi21, t4pi21, e4pi21)}
(V4sc, t4sc, e4sc) = \t{(V4sc, t4sc, e4sc)}

tLongestPulse = {tLongestPulse}, tLongestBSPulse = {tLongestBSPulse}
\t\t\tNLongestBSPulse = {NLongestBSPulse}
T_a34off = {T_a34off}, T_MR = {T_MR}, T_BS = {T_BS}, T_END = {T_END}
T_MR_L = {T_MR_L}, T_MR_R = {T_MR_R}
T_FREE_DELTA_1 = {T_FREE_DELTA_1}, T_FREE_DELTA_2 = {T_FREE_DELTA_2}
N_FREE_STEPS_1 = {N_FREE_STEPS_1}, N_FREE_STEPS_2 = {N_FREE_STEPS_2}
T_BS={T_BS}, T_BS_L={T_BS_L}, T_BS_R={T_BS_R}
""")
# -

v3scat*T_BS_R



if (intensity1 != 1) or (intensity2 != 1): l.warning(f"{intensity1}, {intensity2} should be 1, or some code got messed up (?)")

tbtest = np.arange(0,tLongestPulse,dt)
plt.figure()
plt.plot(tbtest, VS(tbtest, 0.5*t3pi1, t3pi1, V3pi1),label="$\pi$    pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t3pi2, t3pi2, V3pi2),label="$\pi/2$ pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t3sc,  t3sc,  V3sc),label="scat pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4pi1, t4pi1, V4pi1),label="$\pi$    pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4pi2, t4pi2, V4pi2),label="$\pi/2$ pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, 0.5*t4sc,  t4sc,  V4sc),label="scat pulse for ${}^4\mathrm{He}$")
plt.legend()
plt.xlabel("t (ms)")
plt.ylabel("VS")
plt.show()

tbtest = np.arange(0, T_END,dt)
plt.figure()
plt.plot(tbtest, VS(tbtest, 0.5*t4sc,  t4sc,  V4sc),label="scat pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, T_MR, t3pi1, V3pi1),label="$\pi$    pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, T_MR, t4pi1, V4pi1),label="$\pi$    pulse for ${}^4\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, T_BS, t3pi2, V3pi2),label="$\pi/2$ pulse for ${}^3\mathrm{He}$")
plt.plot(tbtest, VS(tbtest, T_BS, t4pi2, V4pi2),label="$\pi/2$ pulse for ${}^4\mathrm{He}$")
plt.axvline(T_a34off,alpha=0.5); #plt.text(T_a34off + 0.01, V4sc,'T_a34off')
plt.axvline(T_MR_L,alpha=0.5)
plt.axvline(T_MR_R,alpha=0.5)
plt.axvline(T_BS_L,alpha=0.5)
plt.axvline(T_BS_R,alpha=0.5)
plt.legend()
plt.xlabel("t (ms)")
plt.ylabel("VS")
pulsetimings = plt.gcf()
plt.show()





# +
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
# -



gc.collect()
# %reset -f in
# %reset -f out



# ### Initial wave function Defs (🔄)

# +
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


# +
# _ = psi0gaussian() # executes in 8.97s 
# -

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



su=5
# _ = psi0_just_opposite(dr=0,s3=su,s4=su,pt=-3*hb*k,a=0*pi) %7sec

# +
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


# +
# _ = psi0Pair() # 5.24s 

# +
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


# +
# _ = psi0ring_with_logging(dr=20,s3=3,s4=3,pt=p,an=1)
# _ = psi0ring(dr=20,s3=3,s4=3,pt=p,an=1) # 13s
# -

gc.collect()
# %reset -f in
# %reset -f out

# + active=""
# sys.getsizeof(psi)
# -



# +
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


# +
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
# -



su = 3
# psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=-100, pz3=-50, px4=100, pz4=50)
# psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0) # 5sec
t = 0

# + vscode={"languageId": "raw"} active=""
# tempTest3 = only3(psi)
# tempTest4 = only4(psi)
# print("check normalisation psi", check_norm(psi))
# phi, swnf = phiAndSWNF(psi)
# tempTest3phi = only3phi(phi)
# tempTest4phi = only4phi(phi)
# print("check normalisation phi", check_norm_phi(phi))
# print("swnf =", swnf)
#
# plt.figure(figsize=(10,6))
# plt.subplot(2,2,1)
# plt.imshow(np.flipud(tempTest3.T), extent=[-xmax,xmax,-zmax,zmax])
#
# plt.subplot(2,2,2)
# plt.imshow(np.flipud(tempTest4.T), extent=[-xmax,xmax,-zmax,zmax])
#
# plt.subplot(2,2,3)
# plt.imshow((tempTest3phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# # plt.colorbar()
#
# plt.subplot(2,2,4)
# plt.imshow((tempTest4phi.T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# # plt.colorbar()
#
# plt.show()
# ram_py_log()
# del tempTest3, tempTest4, tempTest3phi, tempTest4phi
# gc.collect()
# # %reset -f in
# # %reset -f out
# # ram_py_log()
# -





# + active=""
# su = 4
# t = 0
# psi = psi0ring_with_logging(dr=5,s3=1.7,s4=1.7,pt=2*hb*k,an=128) # Takes about 12m
# with pgzip.open(output_prefix+'psi0ring_with_logging_testRun_128'+output_ext,
#                 'wb', thread=8, blocksize=1*10**8) as file:
#     pickle.dump(psi, file) 

# + active=""
# with pgzip.open('/Volumes/tonyNVME Gold/twoParticleSim/20240507-191333-TFF/psi0ring_with_logging_testRun_128.pgz.pkl'
#                 , 'rb', thread=8) as file:
#     psi = pickle.load(file)
# su, t = 4, 0 
# -



# + active=""
# print("check normalisation psi", check_norm(psi))
# phi, swnf = phiAndSWNF(psi)
# print("check normalisation phi", check_norm_phi(phi))
# print("swnf =", swnf)
#
# plt.figure(figsize=(12,6))
# plt.subplot(2,2,1)
# plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax,zmax])
#
# plt.subplot(2,2,2)
# plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax,zmax])
#
# plt.subplot(2,2,3)
# plt.imshow(np.flipud(only3phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# # plt.colorbar()
#
# plt.subplot(2,2,4)
# plt.imshow(np.flipud(only4phi(phi).T), extent=[-pxmax,pxmax,-pzmax,pzmax])
# # plt.colorbar()
#
# plt.show()
# ram_py_log()
# gc.collect()
# %reset -f in
# %reset -f out
# ram_py_log()

# + notebookRunGroups={"groupValue": "2"} vscode={"languageId": "raw"} active=""
# _ = None
# print_ram_usage(globals().items(),10)

# + vscode={"languageId": "raw"} active=""
# plt.figure(figsize=(11,8))
# plt.subplot(2,2,1)
# plt.imshow(np.real(psi[60,:,:,45]))
# plt.subplot(2,2,2)
# plt.imshow(np.real(psi[:,45,60,:].T))
# plt.subplot(2,2,3)
# plt.imshow(np.real(psi[60,:,60,:]))
# plt.subplot(2,2,4)
# plt.imshow(np.real(psi[30,:,90,:]))
# -

su/v3

su/v4



# @njit(cache=True, parallel=True)
@jit(forceobj=True, parallel=True)
def gx3x4_calc(psi,cut=5.0):
    ind = abs(zlin) < (cut+1e-15)*dz
    # print(ind)
    gx3x4 = np.trapz(np.abs(psi[:,:,:,ind])**2,zlin[ind],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind,:],zlin[ind],axis=1)
    return gx3x4


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


# + vscode={"languageId": "raw"} active=""
# plt.figure(figsize=(14,6))
#
# cut_list = [1.0, 10.0, 15.0]
# for i in range(3):
#     cut = cut_list[i]
#     gx3x4 = gx3x4_calc(phi,cut=cut)
#     plt.subplot(1,3,i+1)
#     plot_gx3x4(gx3x4,cut)
# plt.show()
# -









# ## Split-step func

# ### Function Defs (🔄)

# +
# numba.set_num_threads(2)
# Each thread will multiply the memory usage!
# e.g. single thread 10GB -> 2 threads ~30GB
# -

l.info(f"""numba.get_num_threads() = {numba.get_num_threads()}, 
{round(ram_py_GB(),2)} GB used of sys total {round(ram_sys_GB,2)} GB in system
multithread could use up to {round(ram_py_GB()*numba.get_num_threads(),2)} GB !!!""")



# TODO WTF did this come from?
# a34 = 0.029 #µm
# a34 = 0.5 #µm
a34 = sqrt(2)*5*dx
strength34 = 1e5 # I don't know
l.info(f"{-(1j/hb) * strength34 * np.exp(-((0-0)**2 +(0-0)**2)/(4*a34**2)) *0.5*dt}")
l.info(f"a34 = {a34}")

# + active=""
# expContact = np.zeros((nx,nz, nx,nz),dtype=dtypec)
# for (iz3, z3) in enumerate(zlin):
#     for (ix4, x4) in enumerate(xlin):
#         for (iz4, z4) in enumerate(zlin):
#             dis = ((xlin-x4)**2 +(z3-z4)**2)**0.5
#             expContact[:,iz3,ix4,iz4] = np.exp(-(1j/hb) * # this one is unitary time evolution operator
#                                         strength34 *
#                                             0.5*(1+np.cos(2*np.pi/a34*( dis ))) * 
#                                             (-0.5*a34 < dis) * (dis < 0.5*a34)
#                                         # np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2))
#                                                # inside the guassian contact potential
#                                                *0.5*dt
#                                         )  

# + active=""
# plt.figure(figsize=(15,4))
# plt.subplot(1,3,1)
# plt.imshow(np.angle(expContact[:,:,60,45]).T)
# plt.colorbar()
#
# plt.subplot(1,3,2)
# plt.imshow(np.flipud(np.angle(expContact[:,45,:,45]).T))
# plt.colorbar()
#
# plt.subplot(1,3,3)
# plt.imshow(np.flipud(np.angle(expContact[50,:,53,:]).T))
# plt.colorbar()
# plt.show()

# + active=""
# expContact = None
# del expContact
# gc.collect()
# -



# +
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
def scattering_evolve_loop_helper2_inner_phi_step(phi_init,dtsteps=1):
    phi = phi_init
    m1jhb05m3 = -(1j/hb) * (0.5/m3) * (dt*dtsteps)
    m1jhb05m4 = -(1j/hb) * (0.5/m4) * (dt*dtsteps)
    for iz3 in prange(nz):
        pz3 = pzlin[iz3]
#     for (iz3, pz3) in enumerate(pzlin):
        for (ix4, px4) in enumerate(pxlin):
            for (iz4, pz4) in enumerate(pzlin):
                phi[:,iz3,ix4,iz4] *= np.exp(m1jhb05m3 * (pxlin**2 + pz3**2) + \
                                             m1jhb05m4 * (  px4**2 + pz4**2))
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


# +
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
# -

total_steps*dt 



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


output_pre_selp = output_prefix + "scattering_evolve_loop_plot/"
os.makedirs(output_pre_selp, exist_ok=True)
def scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(ram_py_MB(),3)) + "MB  ")

    fig = plt.figure(figsize=(8,7))
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
        plt.savefig(output_pre_selp+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_selp+title+".png", dpi=600, bbox_inches='tight')
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 



output_pre_selpa = output_prefix + "scattering_evolve_loop_plot_alt/"
os.makedirs(output_pre_selpa, exist_ok=True)
def scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=1,po=1):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(ram_py_MB(),3)) + "MB  ")

    fig = plt.figure(figsize=(8,7))
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
        plt.savefig(output_pre_selpa+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_selpa+title+".png", dpi=600, bbox_inches='tight')
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 





evolve_s34 = 1e6
l.info(f"strength34 = {strength34}, \t evolve_s34 = {evolve_s34}")





# ### Initialise Initial State (🈶)

assert False, "just to catch run all"

su = 3
# psi = psi0_just_opposite_double(dr=0,s3=su*(4/3),s4=su,pt=-4.0*hb*k,a=0.5*pi) # 16.37s
psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)
t = 0
f = -1
phi, swnf = phiAndSWNF(psi, nthreads=7)

# + vscode={"languageId": "raw"} active=""
# scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False)

# + vscode={"languageId": "raw"} active=""
# scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=1, po=0.1)
# -

psi_phi_plot2(t,f,psi,phi, titleR="")

_=plot_g34_v2(t, phi)





# + active=""
# scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=10)
# -

_ = None
gc.collect()
# %reset -f in
# %reset -f out
ram_py_log()











# # Simulation Sequence (🈶)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Scattering from perfect init state (🈶)
# -

assert False, "just to catch run all"

evolve_loop_time_start = datetime.now()
_ = scattering_evolve_loop_helper2(t,psi,swnf,steps=1,progress_proxy=None,s34=strength34)
evolve_loop_time_end = datetime.now()
evolve_loop_time_delta = evolve_loop_time_end - evolve_loop_time_start
l.info(f"Time to run one evolve_loop_time_start is {evolve_loop_time_delta} (Run again to use cached compile)")
# need to run this once before looping to cache numba compiles

gc.collect()
# %reset -f in
# %reset -f out
ram_py_log()

print_every = 10
frames_count = 300
total_steps = print_every * frames_count
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""print_every = {print_every}, \tframes_count = {frames_count}, total_steps = {total_steps}
Target simulation end time = {frames_count*print_every*dt} ms
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")

assert False, "just to catch run all"

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
evolve_many_loops_start = datetime.now()
frameAcc = 0 
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        evolve_many_loops_inner_now = datetime.now()
        tE = evolve_many_loops_inner_now - evolve_many_loops_start
        if frameAcc > 0: 
            tR = (frames_count-frameAcc)* tE/frameAcc
            l.info(f"Now l={frameAcc}, t={round(t,6)}, tE={tE}, tR={tR}")
        scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
        scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        gc.collect()
        (t,psi,phi) = scattering_evolve_loop_helper2(t,psi,swnf,
                                                     steps=print_every,progress_proxy=progressbar,s34=evolve_s34)
        frameAcc += 1
f += 1
scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)

l.info(t)
with pgzip.open(output_prefix+f"psi at t={t}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 













# ## Scattering generated from Bragg pulse

# ### Function definitions (🔄)

# +
@njit(parallel=True,cache=True,fastmath=True)
# @njit(parallel=True,cache=True)
def scattering_evolve_bragg_loop_helper2_inner_psi_step(
        psi_init, s34, tnow,
        t3mid, t3wid, v3pot,
        t4mid, t4wid, v4pot,
    ):
    psi = psi_init
    minus1jhb05dt = -(1j/hb)*0.5*dt
    VS3 = VS(tnow,t3mid,t3wid,v3pot)
    VS4 = VS(tnow,t4mid,t4wid,v4pot)
    for iz3 in prange(nz):
        z3 = zlin[iz3]
        for (ix4, x4) in enumerate(xlin):
            for (iz4, z4) in enumerate(zlin):
                # xlin is x3 here in the for loop
                # a34 contact potential
                if s34 != 0:
                    dis = ((xlin-x4)**2 +(z3-z4)**2)**0.5
                    psi[:,iz3,ix4,iz4] *= np.exp(minus1jhb05dt * # this one is unitary time evolution operator
                                                s34 * 0.5*(1+np.cos(2*np.pi/a34*( dis ))) * 
                                                    (-0.5*a34 < dis) * (dis < 0.5*a34)
                                            # np.exp(-((xlin-x4)**2 +(z3-z4)**2)/(4*a34**2)) # inside the guassian contact potential
                                                )
                # Bragg Potential (VxExpGrid in oneParticle)
                if VS3 != 0 or VS4 != 0:
                    psi[:,iz3,ix4,iz4] *= np.exp(minus1jhb05dt * (
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
# @jit(nogil=True, forceobj=True)
@jit(forceobj=True, cache=True)
# @jit(nogil=True)
# @njit(nogil=True, parallel=True)
def scattering_evolve_bragg_loop_helper2(
        tin, psii, swnf, 
        steps=20, progress_proxy=None, s34=strength34,
        t3mid=-2, t3wid=1, v3pot=0,
        t4mid=-2, t4wid=1, v4pot=0,
        numba_threads=7,
        fftw_threads=7
    ):
    t = tin
    psi = psii
    numba.set_num_threads(numba_threads)
    for ia in range(steps):
        # psi = scattering_evolve_bragg_loop_helper2_inner_psi_step(
        #     toPsi(
        #         scattering_evolve_loop_helper2_inner_phi_step(toPhi(
        #             scattering_evolve_bragg_loop_helper2_inner_psi_step(psi,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)
        #             , swnf, nthreads=fftw_threads))
        #         , swnf, nthreads=fftw_threads)
        #     ,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)

        psi = scattering_evolve_bragg_loop_helper2_inner_psi_step(psi,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)
        phi = toPhi(psi, swnf, nthreads=fftw_threads)
        phi = scattering_evolve_loop_helper2_inner_phi_step(phi)
        #del psi  # might cause memory issues
        psi = toPsi(phi, swnf, nthreads=fftw_threads)
        psi = scattering_evolve_bragg_loop_helper2_inner_psi_step(psi,s34,t,t3mid,t3wid,v3pot,t4mid,t4wid,v4pot)
        t += dt 
        if progress_proxy != None:
            progress_proxy.update(1)                                   
    return (t, psi, phi)

@jit(forceobj=True, parallel=True,cache=True)
def evolve_free_part(tin, psii, swnf, dtsteps):
    t = tin
    psi = psii
    phi = toPhi(psi, swnf, nthreads=7)
    phi = scattering_evolve_loop_helper2_inner_phi_step(phi, dtsteps)
    psi = toPsi(phi, swnf, nthreads=7)
    t += dtsteps*dt
    return (t,psi,phi)


# -

# ### One test run (🈶)

evolve_loop_time_start = datetime.now()
_ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,steps=1,progress_proxy=None,s34=evolve_s34,
                                        t3mid=-3,t3wid=1,v3pot=0,
                                        t4mid=0.5*t4sc,t4wid=t4sc,v4pot=V4sc)
evolve_loop_time_end = datetime.now()
evolve_loop_time_delta = evolve_loop_time_end - evolve_loop_time_start
l.info(f"Time to run one evolve_loop_time_start is {evolve_loop_time_delta} (Run again to use cached compile)")
# need to run this once before looping to cache numba compiles
# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Some Dev testing (usually don't run this)

# + active=""
# assert False, "just to catch run all"

# + active=""
# (t,psi,phi) = evolve_free_part(t,psi,swnf,1//dt)

# + active=""
# scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False)
# -



# ### Actually running the fucking thing (🈶)

VS(1*dt ,0.5*t4sc, t4sc, V4sc)

VS(10*dt ,-3,1, V4sc)

gc.collect()
# %reset -f in
# %reset -f out
ram_py_log()
print_ram_usage(globals().items(),10)

T_a34off

print_every = 10 # every us 
frames_count = 600 # until T_a34off
export_every = 30 # every 3us
total_steps = print_every * frames_count
t_end_target = frames_count*print_every*dt
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""print_every = {print_every}, \tframes_count = {frames_count}, total_steps = {total_steps}
Target simulation end time = {t_end_target} ms
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")

assert False, "just to catch run all"

# #### Initial Scattering

# +
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
evolve_many_loops_start = datetime.now()
with ProgressBar(total=total_steps) as progressbar:
    for fl in range(frames_count+1):
        evolve_many_loops_inner_now = datetime.now()
        tP = evolve_many_loops_inner_now - evolve_many_loops_start
        if fl > 0: 
            tR = (frames_count-fl)* tP/fl
            tE = datetime.now() + tR
            l.info(f"Now f={fl}, t={round(t,6)}, tP={tP}, tR={tR}, tE = {tE}")
        
        # scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
        # scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        psi_phi_plot2(t,fl,psi,phi, titleR="sc", skipPlot=True, saveFig=True, settingStr="sc")
        _=plot_g34_v2(t, phi, saveFig=True, skipPlot=True, title2filestr="sc")
        gc.collect()
        if fl % export_every == 0:
            with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
                pickle.dump(psi, file) 

        (t,psi,phi) = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                          steps=print_every,progress_proxy=progressbar,s34=-evolve_s34,
                          ############################################ SCATTERING LENGTH CHANGED TO NEGATIVE !!!!!!
                          t3mid=-3, t3wid=1, v3pot=0, # He3 off 
                          t4mid=0.5*t4sc, t4wid=t4sc, v4pot=V4sc, # He4 scattering       
                          numba_threads=12, fftw_threads=12    
                          )
        
        gc.collect()      

# f += 1
# scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
# scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
psi_phi_plot2(t,fl,psi,phi, titleR="sc", skipPlot=True, saveFig=True, settingStr="sc")
_=plot_g34_v2(t, phi, saveFig=True, skipPlot=True, title2filestr="sc")
with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
                pickle.dump(psi, file)
# -

frameAcc += 1
psi_phi_plot2(t,frameAcc,psi,phi, titleR="", skipPlot=True, saveFig=True)
_=plot_g34_v2(t, phi, saveFig=True, skipPlot=True)

l.info(t)
with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 



t=0.15
with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/20240512-232723-TFF/psi at t={round(t,5)}.pgz.pkl'
                , 'rb', thread=8) as file:
    psi = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=7)
gc.collect()



# #### Free Propagator to Mirror

round(T_MR_L - t_end_target,5)

round((T_MR_L - t_end_target)/dt)

N_FREE_STEPS_1

N_FREE_STEPS_1_actual = round((T_MR_L - t_end_target)/dt - 11)
t_end_target_free_1 = t_end_target + dt*N_FREE_STEPS_1_actual
l.info(f"N_FREE_STEPS_1_actual = {N_FREE_STEPS_1_actual}, t_end_target_free_1 = {t_end_target_free_1}")

t_end_target_free_1

T_MR_L

#### ONE STEP COMPUTE !!!!
(t,psi,phi) = evolve_free_part(t,psi,swnf,N_FREE_STEPS_1_actual)

l.info(t)
with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 

scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)







round(t_end_target_free_1 + tLongestMRPulse,6)

T_MR_R

round((t_end_target_free_1 + tLongestMRPulse)/dt)

# N_MR_STEPS
round((t_end_target_free_1 + tLongestMRPulse)/dt + 11)



# ## Numerical Mirror Pulse (🈶)

ceil(NLongestBSPulse/print_every)

NLongestBSPulse/print_every

T_MR_R

ceil((T_MR_R + 11*dt - t_end_target_free_1)/dt/print_every)





### THIS CELL IS COPIED CODE FROM INITIAL SCATTERING!!!!! 
# print_every = 10 # every us 
# frames_count = ceil(NLongestBSPulse/print_every) # until T_a34off
frames_count = ceil((T_MR_R + 12*dt - t_end_target_free_1)/dt/print_every)
export_every = 20
total_steps = print_every * frames_count
t_end_target = frames_count*print_every*dt
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""print_every = {print_every}, 
frames_count = {frames_count}, 
export_every = {export_every}, \t {frames_count//export_every}
total_steps = {total_steps}
Target simulation end time = {t_end_target} ms, {t_end_target+t}
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")

# +
### THIS CELL IS COPIED CODE FROM INITIAL SCATTERING!!!!! 
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
evolve_many_loops_start = datetime.now()
frameAcc = 0 
with ProgressBar(total=total_steps) as progressbar:
    for f in range(frames_count):
        evolve_many_loops_inner_now = datetime.now()
        tP = evolve_many_loops_inner_now - evolve_many_loops_start
        if frameAcc > 0: 
            tR = (frames_count-frameAcc)* tP/frameAcc
            tE = datetime.now() + tR
            l.info(f"Now f={frameAcc}, t={round(t,6)}, tP={tP}, tR={tR}, tE = {tE}")
        scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
        scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        gc.collect()
        (t,psi,phi) = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                          steps=print_every,progress_proxy=progressbar,s34=evolve_s34,
                          t3mid=T_MR, t3wid=t3pi1, v3pot=V3pi1, #### THESE CHANGED, EVERYTHING ELSE COPIED
                          t4mid=T_MR, t4wid=t4pi1, v4pot=V4pi1  ####                  
                          )
        frameAcc += 1
        if frameAcc % export_every == 0:
            with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
                pickle.dump(psi, file) 
                
f += 1
scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
l.info(f"t={t}")
with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(psi, file) 
gc.collect()
# -





# ## Beam Splitter Pulse

# ### BSP configurations (🔄)

# +
# th3, th3p, th4, th4p sets 
thetaCombo1 = ((0,1),           (V3pi32,t3pi32), (V4pi4, t4pi4),  (V4pi34,t4pi34))
thetaCombo2 = ((V3pi34,t3pi34), (V3pi4, t3pi4),  (V4pi32,t4pi32), (0,1)          )
thetaCombo3 = ((V3pi54,t3pi54), (V3pi34,t3pi34), (V4pi1, t4pi1),  (V4pi32,t4pi32))
thetaCombo4 = ((V3pi32,t3pi32), (V3pi1, t3pi1),  (V4pi34,t4pi34), (V4pi54,t4pi54))

thetaCombo5 = ((V3pi4,t3pi4),   (V3pi74,t3pi74), (0,1),           (V4pi2,t4pi2)  )
thetaCombo6 = ((V3pi2,t3pi2),   (0, 1),          (V4pi74,t4pi74), (V4pi4,t4pi4)  )
# thetaCombo7 = ((V3pi21,t3pi21), (V3pi2,t3pi2),   (V4pi54,t4pi54), (V4pi74,t4pi74))
# thetaCombo8 = ((V3pi74,t3pi74), (V3pi54,t3pi54), (V4pi2,t4pi2),   (V4pi21,t4pi21))
        #     this one is incorrect! 
thetaCombo7 = ((V3pi1,t3pi1),   (V3pi2,t3pi2),   (V4pi54,t4pi54), (V4pi74,t4pi74))
thetaCombo8 = ((V3pi74,t3pi74), (V3pi54,t3pi54), (V4pi2,t4pi2),   (V4pi1,t4pi1)  )


# -

def comboSettingsGen(combo):
    return [(combo[0],combo[2]), (combo[0],combo[3]), 
            (combo[1],combo[2]), (combo[1],combo[3])]


comboSett1 = comboSettingsGen(thetaCombo1)
comboSett2 = comboSettingsGen(thetaCombo2)
comboSett3 = comboSettingsGen(thetaCombo3)
comboSett4 = comboSettingsGen(thetaCombo4)
comboSett5 = comboSettingsGen(thetaCombo5)
comboSett6 = comboSettingsGen(thetaCombo6)
comboSett7 = comboSettingsGen(thetaCombo7)
comboSett8 = comboSettingsGen(thetaCombo8)


def comboSymSwap(combo):
    return (combo[2], combo[3], combo[0], combo[1])


# comboSett1s = comboSettingsGen(comboSymSwap(thetaCombo1))
# comboSett2s = comboSettingsGen(comboSymSwap(thetaCombo2))
# comboSett3s = comboSettingsGen(comboSymSwap(thetaCombo3))
# comboSett4s = comboSettingsGen(comboSymSwap(thetaCombo4))
# comboSett5s = comboSettingsGen(comboSymSwap(thetaCombo5))
# comboSett6s = comboSettingsGen(comboSymSwap(thetaCombo6))
# comboSett7s = comboSettingsGen(comboSymSwap(thetaCombo7))
# comboSett8s = comboSettingsGen(comboSymSwap(thetaCombo8))
thetaCombo1s = ( (V3pi4, t3pi4),  (V3pi34,t3pi34), (0,1),           (V4pi32,t4pi32) )
thetaCombo2s = ( (V3pi32,t3pi32), (0,1)          , (V4pi34,t4pi34), (V4pi4, t4pi4)  )
thetaCombo3s = ( (V3pi1, t3pi1),  (V3pi32,t3pi32), (V4pi54,t4pi54), (V4pi34,t4pi34) )
thetaCombo4s = ( (V3pi34,t3pi34), (V3pi54,t3pi54), (V4pi32,t4pi32), (V4pi1, t4pi1)  )
thetaCombo5s = ( (0,1),           (V3pi2,t3pi2)  , (V4pi4,t4pi4),   (V4pi74,t4pi74) )
thetaCombo6s = ( (V3pi74,t3pi74), (V3pi4,t3pi4)  , (V4pi2,t4pi2),   (0, 1)          )
thetaCombo7s = ( (V3pi54,t3pi54), (V3pi74,t3pi74), (V4pi1,t4pi1),   (V4pi2,t4pi2)   )
thetaCombo8s = ( (V3pi2,t3pi2),   (V3pi1,t3pi1)  , (V4pi74,t4pi74), (V4pi54,t4pi54) )
comboSett1s = comboSettingsGen(thetaCombo1s)
comboSett2s = comboSettingsGen(thetaCombo2s)
comboSett3s = comboSettingsGen(thetaCombo3s)
comboSett4s = comboSettingsGen(thetaCombo4s)
comboSett5s = comboSettingsGen(thetaCombo5s)
comboSett6s = comboSettingsGen(thetaCombo6s)
comboSett7s = comboSettingsGen(thetaCombo7s)
comboSett8s = comboSettingsGen(thetaCombo8s)

comboSettSet = [comboSett1, comboSett2, comboSett3, comboSett4, comboSett5, comboSett6, comboSett7, comboSett8, 
                comboSett1s, comboSett2s, comboSett3s, comboSett4s, comboSett5s, comboSett6s, comboSett7s, comboSett8s]

T_BS_L



gridX3, gridX4 = np.meshgrid(xlin,xlin)
@jit(forceobj=True, cache=True)
def plot_bs_l_free_time(psiH, tt, midCutZ=[1,5], pltShow=True, pltSave=True):
    # midCutZ = 5
    plt.figure(figsize=(7*len(midCutZ),11))
    tempReturn = []
    tempReturn2 = []
    for mi, mc in enumerate(midCutZ):
        plt.subplot(2,len(midCutZ),mi+1)
        ind = abs(zlin-0) < mc
        # l.info(f"np.sum(ind) = {np.sum(ind)}")
        midCut = np.trapz(np.abs(psiH[:,:,:,ind])**2,zlin[ind],axis=3)
        midCut = np.trapz(midCut[:,ind,:],zlin[ind],axis=1)
        tempReturn2.append(midCut)
        # plt.imshow(np.log(np.flipud(midCut.T)),cmap="Greens",extent=[-xmax,+xmax,-zmax,+zmax])
        plt.imshow(np.flipud(midCut.T)**0.5,cmap="Greens",extent=[-xmax,+xmax,-zmax,+zmax])
        plt.title(f"$t$={tt}, $dz$={mc}, ($s$={np.sum(ind)})")
        # plt.colorbar()
        plt.xlabel("$x3$")
        plt.ylabel("$x4$")
        plt.axhline(y=0,color='k',alpha=0.2,linewidth=0.7)
        plt.axvline(x=0,color='k',alpha=0.2,linewidth=0.7)

        plt.subplot(2,len(midCutZ),mi+len(midCutZ)+1)
        tempDiagIntensity = np.zeros(nx)
        for (ind, x0filt) in enumerate(xlin):
            # gridFilter = np.exp(-((gridX3-x0filt)**2 + (gridX4-(-4/3)*x0filt)**2)/mc )
            gridFilter = np.exp(-((gridX3-(-3/4)*x0filt)**2 + (gridX4-x0filt)**2)/mc )
            tempDiagIntensity[ind] = np.sum(midCut*gridFilter)
        plt.axvline(x=+v3scat*tt,alpha=0.2,color='g')
        plt.axvline(x=-v3scat*tt,alpha=0.2,color='g')
        plt.axvline(x=+v3scat*(tt-t4sc),alpha=0.2,color='g')
        plt.axvline(x=-v3scat*(tt-t4sc),alpha=0.2,color='g')
        plt.axvline(x=+v3scat*(tt-tLongestPulse),alpha=0.2,color='g')
        plt.axvline(x=-v3scat*(tt-tLongestPulse),alpha=0.2,color='g')
        plt.plot(xlin,tempDiagIntensity,)

        tempReturn.append(tempDiagIntensity)
    if pltSave:
        title = f"BS_L free time at t={round(tt,5)}, dz={midCutZ}"
        plt.savefig(output_prefix+title+".png",dpi=600)
        plt.savefig(output_prefix+title+".pdf",dpi=600)
    if pltShow:
        plt.show()
    plt.close()
    return (tempReturn,tempReturn2)


# ### Loading previous psi (🈶)

assert False, "just to catch run all"

t=0.6347
# data_folder = "20240521-231755-TFF"
data_folder = "20240711-234819-TFF"
with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl'
                , 'rb', thread=8) as file:
    psi = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=7)
gc.collect()

t=0.6347
T_FREE_TO_BS = T_BS_L - 5*dt - t
# T_FREE_TO_BS = 1.0 - t
N_FREE_TO_BS = ceil(T_FREE_TO_BS/dt)
l.info(f"""T_FREE_TO_BS = {T_FREE_TO_BS}
t+T_FREE_TO_BS = {t+T_FREE_TO_BS}
N_FREE_TO_BS = {N_FREE_TO_BS}""")

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ##### One Test Run
# -

(t2,psi2,phi2) = evolve_free_part(t,psi,swnf,N_FREE_TO_BS)

scattering_evolve_loop_plot(t2,-1,psi2,phi2, plt_show=True, plt_save=False)

scattering_evolve_loop_plot_alt(t2,-1,psi2,phi2, plt_show=True, plt_save=False, logPlus=1, po=0.1)

gc.collect()
# %reset -f in
# %reset -f out
ram_py_log()

# + active=""
# gridX3, gridX4 = np.meshgrid(xlin,xlin)
# widthFilter = 1
# gridFilter = np.exp(-( (3/5)*gridX + (-4/5)*gridZ )**2 / widthFilter  )
# x0filt = -28
# gridFilter = np.exp(-((gridX3-(-3/4)*x0filt)**2 + (gridX4-x0filt)**2)/ widthFilter )
# plt.imshow(gridFilter)
# plt.imshow(np.flipud(gridFilter.T),cmap="Greens",extent=[-xmax,+xmax,-xmax,+xmax])
# plt.show()
# -

# plot_bs_l_free_time(psi2, t2, midCutZ=[1,3,10], pltShow=True, pltSave=True)
_ = plot_bs_l_free_time(psi2, t2, midCutZ=[1,2,5], pltShow=True, pltSave=False)

# + active=""
# gridX3, gridX4 = np.meshgrid(xlin,xlin)
# widthFilter = 1
# # gridFilter = np.exp(-( (3/5)*gridX + (-4/5)*gridZ )**2 / widthFilter  )
# x0filt = +10
# gridFilter = np.exp(-((gridX3-x0filt)**2 + (gridX4-(-4/3)*x0filt)**2)/ widthFilter  )
# plt.imshow(np.flipud(gridFilter.T),cmap="Greens",extent=[-xmax,+xmax,-xmax,+xmax])
# plt.show()
# -







# + [markdown] jp-MarkdownHeadingCollapsed=true
# ##### BS Timing Scan?

# + active=""
# plt.figure(pulsetimings.number)
# # pulsetimings
# # plt.axvline(T_BS)
# plt.show()

# + active=""
# T_BS_SCAN_DELTA = 3*tLongestBSPulse

# + active=""
# T_BS_SCAN_DT_SKIPS = 30
# 2*T_BS_SCAN_DELTA / (T_BS_SCAN_DT_SKIPS*dt)
# -

np.array([T_BS, T_BS_L, T_BS_R, tLongestBSPulse]) * 1000

(T_BS-1.5*tLongestBSPulse, T_BS+0.5*tLongestBSPulse)

# T_BS_L_SCAN_TARGETS = np.arange(T_BS-1.5*tLongestBSPulse, T_BS+0.5*tLongestBSPulse, 50*dt)
T_BS_L_SCAN_TARGETS = np.arange(1.00,1.30, 20*dt)
l.info(f"""len(T_BS_L_SCAN_TARGETS) = {len(T_BS_L_SCAN_TARGETS)}
T_BS_L_SCAN_TARGETS: {T_BS_L_SCAN_TARGETS}""")

t_bs_scan_diag_result = []
t_bs_scan_diag_result2 = []
for (ind, tbs) in tqdm(enumerate(T_BS_L_SCAN_TARGETS),total=len(T_BS_L_SCAN_TARGETS)):
    N_FREE_TO_BS = ceil((tbs-t)/dt)
    (t2,psi2,phi2) = evolve_free_part(t,psi,swnf,N_FREE_TO_BS)
    scattering_evolve_loop_plot(t2,ind,psi2,phi2, plt_show=False, plt_save=True)
    scattering_evolve_loop_plot_alt(t2,ind,psi2,phi2, plt_show=False, plt_save=True, logPlus=1, po=0.1)
    (tbs_result, tbs_resul2) = plot_bs_l_free_time(psi2, t2, midCutZ=[1,2,5], pltShow=False, pltSave=True)
    t_bs_scan_diag_result.append(tbs_result)
    t_bs_scan_diag_result2.append(tbs_resul2)
    gc.collect()
    # print(N_FREE_TO_BS)

with pgzip.open(output_prefix+f"t_bs_scan_diag_result"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(t_bs_scan_diag_result, file) 
with pgzip.open(output_prefix+f"t_bs_scan_diag_result2"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(t_bs_scan_diag_result2, file) 

gc.collect()





T_MR

T_MR*2*v3

T_MR*2*v4

m3/(m3+m4)*v4*(2*T_MR) #v4

m4/(m3+m4)*v4*(2*T_MR) #v3

t_bs_scan_diag_result_bk = t_bs_scan_diag_result

t_bs_scan_diag_result_np = np.array(t_bs_scan_diag_result)

((nx-1)//2-middle_mask,(nx-1)//2+1+middle_mask)

# +
mask = np.ones(t_bs_scan_diag_result_np.shape[1], dtype=bool)
middle_mask = 6
mased_bs_scan = None
mased_bs_scan = np.copy(t_bs_scan_diag_result_np[:,1])
# mased_bs_scan[:,(nx-1)//2-middle_mask : (nx-1)//2+1+middle_mask] = 0

plt.imshow(np.flipud(mased_bs_scan)**0.001, 
# plt.imshow(np.flipud(t_bs_scan_diag_result_np[:,0]), 
           extent=[-xmax,xmax,T_BS_L_SCAN_TARGETS[0],T_BS_L_SCAN_TARGETS[-1]],
           aspect=2*xmax/(T_BS_L_SCAN_TARGETS[-1]-T_BS_L_SCAN_TARGETS[0]), cmap="Greens"
          )
plt.colorbar()
plt.show()
# -

T_BS

mask

filtered_data



10*dx

t_bs_scan_diag_result_np.shape

output_prefix

psixeqList = []
psiyeqList = []
for (ind, tbs) in enumerate(T_BS_L_SCAN_TARGETS):
    midCut = t_bs_scan_diag_result2[ind]
    psixeq = np.trapz(midCut[0], xlin, axis=0)
    psiyeq = np.trapz(midCut[0], zlin, axis=1)
    psixeqList.append(psixeq)
    psiyeqList.append(psiyeq)
psixeqList = np.array(psixeqList)
psiyeqList = np.array(psiyeqList)

# + active=""
# plt.imshow(np.flipud(midCut[0].T))

# + active=""
# plt.plot(psixeq)

# + active=""
# plt.plot(psixeqList[0,:])
# -

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(np.flipud(psixeqList)**0.001,
           extent=[-xmax,xmax, T_BS_L_SCAN_TARGETS[0], T_BS_L_SCAN_TARGETS[-1]],
           aspect= 4*xmax/(T_BS_L_SCAN_TARGETS[-1]-T_BS_L_SCAN_TARGETS[0])
          )
plt.xlabel("$x_4$")
plt.yticks(np.linspace(T_BS_L_SCAN_TARGETS[0],T_BS_L_SCAN_TARGETS[-1], 31))
plt.subplot(1,2,2)
plt.imshow(np.flipud(psiyeqList)**0.001,
           extent=[-xmax,xmax, T_BS_L_SCAN_TARGETS[0], T_BS_L_SCAN_TARGETS[-1]],
           aspect= 4*xmax/(T_BS_L_SCAN_TARGETS[-1]-T_BS_L_SCAN_TARGETS[0])
          )
plt.xlabel("$x_3$")
plt.show()

1.1*v4scat

1.1*v3scat



# ##### Actual BS Pulses

# +
# may 22, oh fuck it, I'm just going to run with whatever
# -

del psi2, phi2

t=0.6347

(t,psi,phi) = evolve_free_part(t,psi,swnf,N_FREE_TO_BS)

###0 THIS CELL IS COPIED CODE FROM INITIAL SCATTERING!!!!! 
t=0.6347 + N_FREE_TO_BS*dt
print_every = 10 # every us 
# frames_count = ceil(NLongestBSPulse/print_every) # until T_a34off
frames_count = ceil((T_BS_R + 11*dt - t)/dt/print_every)
export_every = 30
total_steps = print_every * frames_count
t_end_target = frames_count*print_every*dt
evolve_loop_time_estimate = total_steps*evolve_loop_time_delta*1.1
l.info(f"""t={t},
print_every = {print_every}, \tframes_count = {frames_count}, total_steps = {total_steps}
Target simulation end time = {t_end_target} ms, {round(t+t_end_target,6)}
Estimated script runtime = {evolve_loop_time_estimate} which is {datetime.now()+evolve_loop_time_estimate}""")

N_FREE_TO_BS*dt

T_END

t3pi1

# +
# evolve_loop_time_delta = datetime.timedelta(seconds=1)
# -

len(comboSettSet)

comboSettSetS = set()
for combSetH in comboSettSet:
    print(combSetH)
    for tCombo in combSetH:
        (vv3, tt3), (vv4, tt4) = tCombo 
        # print(((vv3, tt3), (vv4, tt4)))
        # print(((round(vv3), round(tt3,6)), (round(vv4), round(tt4,6))))
        # print(((round(vv3/VR/0.02), round(tt3,6)), (round(vv4/VR/0.015), round(tt4,6))))
        # print(f"{round(vv3/VR/0.02)}π/4, {round(vv4/VR/0.015)}π/4")
        # print(f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}")
        settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
        comboSettSetS.add(settingStr)
        print(settingStr)

thispklfile

os.path.exists(thispklfile)

gc.collect()
print_ram_usage()

gc.collect()
for cind, combSetH in enumerate(comboSettSet):
    l.info(f"Combo Set ind = {cind}")
    for tCombo in combSetH:
        t=1.2052
        (vv3, tt3), (vv4, tt4) = tCombo 
        # IDEA: use 1/16 division scans? and scan all compo rather than just bell angles
        # IDEA: also need to scan T_BS using corrE as signal.
        # print(((vv3, tt3), (vv4, tt4)))
        # settingStr = ((round(vv3), round(tt3,6)), (round(vv4), round(tt4,6)))
        settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
        # l.info(settingStr)
        l.info(f"  Setting: {round(vv3/VR/0.02)}π/4, {round(vv4/VR/0.015)}π/4, \t ShortForm {settingStr}")
        thispklfile = output_prefix+f"psi at t={round(t,6)} s={settingStr}"+output_ext
        if os.path.exists(thispklfile): 
            l.info(f"    Skipped: {thispklfile}")
            continue
        
        l.info("    Calculating...")
        t=0.6347
        data_folder = "20240919-173152-TFF"#"20240711-234819-TFF" # "20240521-231755-TFF"
        psi = None
        phi = None
        gc.collect()
        with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl'
                        , 'rb', thread=8) as file:
            psi = pickle.load(file)
        phi, swnf = phiAndSWNF(psi, nthreads=7)
        gc.collect()
        (t,psi,phi) = evolve_free_part(t,psi,swnf,N_FREE_TO_BS)
        
        ### THIS CELL IS COPIED CODE FROM INITIAL SCATTERING!!!!! 
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        
        evolve_many_loops_start = datetime.now()
        frameAcc = 0 
        with ProgressBar(total=total_steps) as progressbar:
            for f in range(frames_count):
                evolve_many_loops_inner_now = datetime.now()
                tP = evolve_many_loops_inner_now - evolve_many_loops_start
                # if frameAcc > 0: 
                #     tR = (frames_count-frameAcc)* tP/frameAcc
                #     tE = datetime.now() + tR
                #     l.info(f"Now f={frameAcc}, t={round(t,6)}, tP={tP}, tR={tR}, tE = {tE}")
                # scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
                # scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
                gc.collect()
                (t,psi,phi) = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                                  steps=print_every,progress_proxy=progressbar,s34=0,
                                  t3mid=T_BS, t3wid=tt3, v3pot=vv3, #### THESE CHANGED, EVERYTHING ELSE COPIED
                                  t4mid=T_BS, t4wid=tt4, v4pot=vv4, ####     
                                  numba_threads = 8,
                                  fftw_threads  = 8             
                                  )
                frameAcc += 1
                # if frameAcc % export_every == 0:
                #     with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                #         'wb', thread=8, blocksize=1*10**8) as file:
                #         pickle.dump(psi, file) 
                        
        f += 1
        # scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
        # scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        phi = None
        with pgzip.open(thispklfile,
                        'wb', thread=8, blocksize=1*10**8) as file:
            pickle.dump(psi, file) 
        gc.collect()

time.sleep(3)

# +
# numba.config.NUMBA_NUM_THREADS

# +
# numba.set_num_threads(6)
# -

numba.get_num_threads()





settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
thispklfile = output_prefix+f"psi at t={round(t,6)} s={settingStr}"+output_ext
os.path.exists(thispklfile)

os.path.exists(output_prefix+f"psi at t={round(0.6347,6)}"+output_ext)

# ##### Massive everything scans

omegaNDivs = 8
omegaNDivsList = np.arange(0,2*omegaNDivs+1)
# omegaNDivsList = np.flip(omegaNDivsList)
omegalist = pi/omegaNDivs*omegaNDivsList
l.info(f"omegaNDivsList = {omegaNDivsList}")
l.info(f"omegalist = {omegalist}")

frames_count

print_every

2%2

# + notebookRunGroups={"groupValue": "2"}
(2%omegaNDivs)
# -

int((V4pi1*5/omegaNDivs)/VR/0.015*(omegaNDivs/4))

# +
for (i3,o3) in enumerate(omegaNDivsList):
    for (i4,o4) in enumerate(omegaNDivsList):
        # if i3>=1: break # DEBUGGING SKIP LOOP
        
        t=1.2052
        # (vv3, tt3), (vv4, tt4) = tCombo 
        # print(((vv3, tt3), (vv4, tt4)))
        # settingStr = ((round(vv3), round(tt3,6)), (round(vv4), round(tt4,6)))
        tt3 = t3pi1
        tt4 = t4pi1
        vv3 = V3pi1 * o3/omegaNDivs
        vv4 = V4pi1 * o4/omegaNDivs
        sv3 = round(vv3/VR/0.02 *(omegaNDivs/4))
        sv4 = round(vv4/VR/0.015*(omegaNDivs/4))
        assert sv3 == o3 and sv4 == o4, "I think this should be fine?"
        settingStr = f"{sv3}-{sv4}"
        print(f"i3={i3}, i4={i4}, o3={o3}, o4={o4}, sv3={sv3}, sv4={sv4}") 
        # l.info(settingStr)
        if omegaNDivs == 4 or (sv3%(omegaNDivs/4)==0 and sv4%(omegaNDivs/4)==0):
            settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
            thispklfile = output_prefix+f"psi at t={round(t,6)} s={settingStr}"+output_ext
            thisPhiCorrFile = output_prefix+f"phi-corr-stats s={settingStr}"+output_ext
        else:
            thispklfile = output_prefix+f"psi at t={round(t,6)} s={settingStr}D{omegaNDivs}"+output_ext
            thisPhiCorrFile = output_prefix+f"phi-corr-stats s={settingStr}D{omegaNDivs}"+output_ext

        l.info(f"  Setting: {sv3}π/{omegaNDivs}, {sv4}π/{omegaNDivs}, \t path: {thispklfile}")
        if os.path.exists(thispklfile): 
            l.info(f"    File exist, SKIPPED (yey)")
            continue
        

        l.info("    Calculating...")
        t=0.6347
        data_folder = "20240919-173152-TFF"#"20240919-173152-TFF" #"20240711-234819-TFF" # "20240521-231755-TFF"
        psi = None
        phi = None
        gc.collect()

        # continue # DEBUGGING SKIP LOOP SPEED UP
        with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=8) as file:
            psi = pickle.load(file)
        phi, swnf = phiAndSWNF(psi, nthreads=7)
        gc.collect()
        (t,psi,phi) = evolve_free_part(t,psi,swnf,N_FREE_TO_BS)
        
        ### THIS CELL IS COPIED CODE FROM INITIAL SCATTERING!!!!! 
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        
        evolve_many_loops_start = datetime.now()
        frameAcc = 0 
        with ProgressBar(total=total_steps) as progressbar:
            for f in range(frames_count):
                evolve_many_loops_inner_now = datetime.now()
                tP = evolve_many_loops_inner_now - evolve_many_loops_start
                # if frameAcc > 0: 
                #     tR = (frames_count-frameAcc)* tP/frameAcc
                #     tE = datetime.now() + tR
                #     l.info(f"Now f={frameAcc}, t={round(t,6)}, tP={tP}, tR={tR}, tE = {tE}")
                # scattering_evolve_loop_plot(t,f,psi,phi, plt_show=False, plt_save=True)
                # scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
                gc.collect()
                (t,psi,phi) = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                                  steps=print_every,progress_proxy=progressbar,s34=0,
                                  t3mid=T_BS, t3wid=tt3, v3pot=vv3, #### THESE CHANGED, EVERYTHING ELSE COPIED
                                  t4mid=T_BS, t4wid=tt4, v4pot=vv4,  ####      
                                  numba_threads = 8,
                                  fftw_threads  = 8              
                                  )
                frameAcc += 1
                # t += print_every * dt
                # if frameAcc % export_every == 0:
                #     with pgzip.open(output_prefix+f"psi at t={round(t,6)}"+output_ext,
                #         'wb', thread=8, blocksize=1*10**8) as file:
                #         pickle.dump(psi, file) 
                        
        f += 1
        l.info(f"    t = {t}")
        # scattering_evolve_loop_plot(t,f+1,psi,phi, plt_show=False, plt_save=True)
        # scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)


        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        scattering_evolve_loop_plot(t,settingStr,psi,phi, plt_show=False, plt_save=True)
        scattering_evolve_loop_plot_alt(t,settingStr,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
        psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",cmax3x=2e-3,cmax4x=2e-3,cmax3p=2e-5,cmax4p=2e-5)
        psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",cmax3x=5e-3,cmax4x=5e-3,cmax3p=5e-5,cmax4p=5e-5)
        psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",cmax3x=1e-3,cmax4x=1e-3,cmax3p=1e-5,cmax4p=1e-5)
        tempPlotOutput = plot_g34_v2(phi, cutPlot=1.5, saveFig=True, pMaxCut=2,title2=f"$\phi_3$={round(vv3/VR/0.02)}π/4, $\phi_4=${round(vv4/VR/0.015)}π/4",title2filestr=settingStr, skipPlot=True)
        with pgzip.open(thisPhiCorrFile,'wb', thread=8, blocksize=1*10**8) as file:
            pickle.dump(tempPlotOutput, file)
        l.info(f"  calced corrE({settingStr}) = {corrE(np.flipud(tempPlotOutput[4].T))}")
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 


        phi = None 
        with pgzip.open(thispklfile,
                        'wb', thread=8, blocksize=1*10**8) as file:
            pickle.dump(psi, file) 
        gc.collect()
    


# -

for (i3,o3) in enumerate(omegaNDivsList):
    for (i4,o4) in enumerate(omegaNDivsList):
        print(i3,o3,i4,o4)
        print(f"  {o3+o4}\t {(o3+o4)*2*pi/omegaNDivs}")



# # Some Correlation Calculations

# ### Function Definitations (🔄)

# @jit(forceobjd=True, parallel=True)
def gp3p4_dhalo_calc_noAb(phi,cut=5.0,offset3=0,offset4=0):
    ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
    ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
    gx3x4 = np.trapz(phi[:,:,:,ind4],pzlin[ind4],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind3,:],pzlin[ind3],axis=1)
    # print(gx3x4.shape)
    return gx3x4 
def gp3p4_dhalo_calc(phiHere,cut=5.0,offset3=0,offset4=0):
    ind3 = abs(pzlin-offset3) < (cut+1e-15)*dpz
    ind4 = abs(pzlin-offset4) < (cut+1e-15)*dpz
    gx3x4 = np.trapz(np.abs(phiHere[:,:,:,ind4])**2,pzlin[ind4],axis=3)
    gx3x4 = np.trapz(gx3x4[:,ind3,:],pzlin[ind3],axis=1)
    xip = pxlin > +0*cut*dpz 
    xim = pxlin < -0*cut*dpz 
    #REVIEW: proper integration box
    gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
    return (gx3x4,[gpp,gpm,gmp,gmm])


def gp3p4_dhaloUD_calc(phi,cut=5.0,offset3=0,offset4=0):
    ind3 = abs(pxlin-offset3) < (cut+1e-15)*dpz
    ind4 = abs(pxlin-offset4) < (cut+1e-15)*dpz
    # print(ind4)
    gx3x4 = np.trapz(np.abs(phi[:,:,ind4,:])**2,pxlin[ind4],axis=2)
    gx3x4 = np.trapz(gx3x4[ind3,:,:],pxlin[ind3],axis=0)
    xip = pxlin > +0*cut*dpz 
    xim = pxlin < -0*cut*dpz 
    gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
    return (gx3x4,[gpp,gpm,gmp,gmm])


def plot_dhalo_gp3p4(gx3x4,cut,offset3=0,offset4=0):
    xip = pxlin > +0*cut*dpz 
    xim = pxlin < -0*cut*dpz 
    gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
    E = (gpp+gmm-gpm-gmp)/((gpp+gmm+gpm+gmp))
    
    plt.imshow(np.flipud(gx3x4.T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Greens')
    plt.title("$g^{(2)}_{\pm\pm}$ of $p_\mathrm{cut} = "+str(cut)+"dpz$ and $E="+str(round(E,4))+"$")
    plt.xlabel("$p_3$")
    plt.ylabel("$p_4$")
    plt.axhline(y=0,color='k',alpha=0.2,linewidth=0.7)
    plt.axvline(x=0,color='k',alpha=0.2,linewidth=0.7)
    plt.text(+pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{++}="+str(round(gpp,1))+"$", color='k',ha='center',alpha=0.9)
    plt.text(-pxmax*0.6/(hb*k),+pxmax*0.8/(hb*k),"$g^{(2)}_{-+}="+str(round(gmp,1))+"$", color='k',ha='center',alpha=0.9)
    plt.text(+pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{+-}="+str(round(gpm,1))+"$", color='k',ha='center',alpha=0.9)
    plt.text(-pxmax*0.6/(hb*k),-pxmax*0.8/(hb*k),"$g^{(2)}_{--}="+str(round(gmm,1))+"$", color='k',ha='center',alpha=0.9)


# +
# https://artmenlope.github.io/plotting-complex-variable-functions/
def colorize(fz):
    """
    The original colorize function can be found at:
    https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
    by the user nadapez.
    """
    r = np.log2(1. + np.abs(fz))
    h = np.angle(fz)/(2*np.pi)
    # l = 1 - 0.45**(np.log(1+r)) 
    # l = 0.75*((np.abs(fz))/np.abs(fz).max())**1.2
    l = ((np.abs(fz)**2)/((np.abs(fz)**2).max()))
    s = 1
    c = np.vectorize(hls_to_rgb)(h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (m,n,3)
    c = np.rot90(c.transpose(2,1,0), 1) # Change shape to (m,n,3) and rotate 90 degrees
    return c

legend_elements = [Line2D([0], [0], marker='o', color='cyan', label='$Arg=\pm\pi$', markersize=10, lw=0),
                   Line2D([0], [0], marker='o', color='red', label='$Arg=0$', markersize=10, lw=0)]
# -


output_pre_corr = output_prefix + "corrPlot/"
os.makedirs(output_pre_corr, exist_ok=True)


def plot_g34(phiHere, cutPlot=1.5, saveFig=True, 
             title2="", title2filestr="NA",
             skipPlot=False):
    gx3px4p = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=+p,offset4=+p)
    gx3px4m = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=+p,offset4=-p)
    gx3mx4p = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=-p,offset4=+p)
    gx3mx4m = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=-p,offset4=-p)
    
    gx3x4combined = np.zeros((2*nx,2*nx))
    gx3x4combined[:nx, :nx] = gx3px4p[0]
    gx3x4combined[:nx, nx:] = gx3px4m[0]
    gx3x4combined[nx:, :nx] = gx3mx4p[0]
    gx3x4combined[nx:, nx:] = gx3mx4m[0]

    gx3x4combined2 = np.zeros((2*2,2*2))
    gx3x4combined2[:2, :2] = [[gx3px4p[1][3],gx3px4p[1][2]],[gx3px4p[1][1],gx3px4p[1][0]]]
    gx3x4combined2[:2, 2:] = [[gx3px4m[1][3],gx3px4m[1][2]],[gx3px4m[1][1],gx3px4m[1][0]]]
    gx3x4combined2[2:, :2] = [[gx3mx4p[1][3],gx3mx4p[1][2]],[gx3mx4p[1][1],gx3mx4p[1][0]]]
    gx3x4combined2[2:, 2:] = [[gx3mx4m[1][3],gx3mx4m[1][2]],[gx3mx4m[1][1],gx3mx4m[1][0]]]
    gx3x4n = gx3x4combined2/sum(sum(gx3x4combined2))
    
    # if not skipPlot: 
    ticks = np.linspace(0, 2*nx, 17)
    ticksL = np.linspace(0, 2*nx, 9)
    tick_labels = ["","+3","+2","+1","0","-1","-2","-3","","+3","+2","+1","0","-1","-2","-3",""]
    # tick_labels = np.concatenate((np.linspace(-4, 4, 5), np.linspace(-4, 4, 5)))
    # plt.imshow(np.flipud(gx3x4combined.T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Greens')
    plt.figure(figsize=(11,5))
    ax = plt.subplot(1,2,1)
    im = ax.imshow(np.flipud(gx3x4combined.T),cmap='Greens')
    # ax.set_yticks(ticksL, ["","A↗︎","","A↖︎","","A↘︎","","A↙︎",""])
    # ax.set_xticks(ticksL, ["","B↗︎","","B↖︎","","B↘︎","","B↙︎",""])
    ax.set_yticks(ticks, ["","","","A↗︎","","A↖︎","","","","","","A↘︎","","A↙︎","","",""])
    ax.set_xticks(ticks, ["","","","B↙︎","","B↘︎","","","","","","B↖︎","","B↗","","",""])
    
    ax.axhline(y=1.0*nx,color='k',alpha=0.3,linewidth=0.7)
    ax.axhline(y=0.5*nx,color='k',alpha=0.1,linewidth=0.7)
    ax.axhline(y=1.5*nx,color='k',alpha=0.1,linewidth=0.7)
    ax.axvline(x=1.0*nx,color='k',alpha=0.3,linewidth=0.7)
    ax.axvline(x=0.5*nx,color='k',alpha=0.1,linewidth=0.7)
    ax.axvline(x=1.5*nx,color='k',alpha=0.1,linewidth=0.7)
    ax2 = ax.secondary_xaxis('top')
    ax3 = ax.secondary_yaxis('right')
    ax2.set_xticks(ticks, tick_labels[::-1])
    ax3.set_yticks(ticks, tick_labels)
    plt.title(f"t = {t}")
    
    # l.info(f"""gx3px4p[1] = {gx3px4p[1]}
    # gx3px4m[1] = {gx3px4m[1]}
    # gx3mx4p[1] = {gx3mx4p[1]}
    # gx3mx4m[1] = {gx3mx4m[1]}""")
    
    ax = plt.subplot(1,2,2)
    im = ax.imshow(np.flipud(gx3x4combined2.T),cmap='Greens')
    ticks=np.arange(0,4,1)
    ax.set_yticks(ticks, ["A↗︎","A↖︎","A↘︎","A↙︎"])
    ax.set_xticks(ticks, ["B↙︎","B↘︎","B↖︎","B↗"])
    # ax2 = ax.secondary_xaxis('top')
    # ax3 = ax.secondary_yaxis('right')
    # ax2.set_yticks(ticks, ["A↗︎","A↖︎","A↘︎","A↙︎"])
    # ax3.set_xticks(ticks, ["B↙︎","B↘︎","B↖︎","B↗"])
    for i in range(gx3x4n.shape[0]):
        for j in range(gx3x4n.shape[1]):
            plt.text(j, i, str(round(np.flipud(gx3x4n.T)[i, j],4)), ha='center', va='center', color='dodgerblue')
    plt.title(f"{title2}\ncut = {cutPlot}")
    title = f"CorrE t={round(t,5)}, cut = {cutPlot}, s={title2filestr}"
    if saveFig:
        plt.savefig(output_pre_corr+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_corr+title+".png", dpi=600, bbox_inches='tight')
    if skipPlot:
        plt.close()
    else:
        plt.show()
    return (gx3px4p, gx3px4m, gx3mx4p, gx3mx4m, gx3x4n)


def plot_g34_v2(t, phi, cutPlot=1.5, saveFig=True, 
             pMaxCut=2,
             title2="", title2filestr="NA",
             skipPlot=False,
             figSize=(9,3.0),
             sws=0.15
             ):
    gx3px4p = gp3p4_dhalo_calc(phi,cut=cutPlot,offset3=+p,offset4=+p)
    gx3px4m = gp3p4_dhalo_calc(phi,cut=cutPlot,offset3=+p,offset4=-p)
    gx3mx4p = gp3p4_dhalo_calc(phi,cut=cutPlot,offset3=-p,offset4=+p)
    gx3mx4m = gp3p4_dhalo_calc(phi,cut=cutPlot,offset3=-p,offset4=-p)
    
    nc = int(pMaxCut*p/dpx)*2+1;
    nr = (nx-nc)//2
    # print(nc,nr)
    # print(gx3px4p[0].shape)
    # print(gx3px4p[0][nr:-nr,nr:-nr].shape)

    gx3x4combined = np.zeros((2*nc,2*nc))
    gx3x4combined[:nc, :nc] = gx3px4p[0][nr:-nr,nr:-nr]
    gx3x4combined[:nc, nc:] = gx3px4m[0][nr:-nr,nr:-nr]
    gx3x4combined[nc:, :nc] = gx3mx4p[0][nr:-nr,nr:-nr]
    gx3x4combined[nc:, nc:] = gx3mx4m[0][nr:-nr,nr:-nr]

    gx3x4combined2 = np.zeros((2*2,2*2))
    gx3x4combined2[:2, :2] = [[gx3px4p[1][3],gx3px4p[1][2]],[gx3px4p[1][1],gx3px4p[1][0]]]
    gx3x4combined2[:2, 2:] = [[gx3px4m[1][3],gx3px4m[1][2]],[gx3px4m[1][1],gx3px4m[1][0]]]
    gx3x4combined2[2:, :2] = [[gx3mx4p[1][3],gx3mx4p[1][2]],[gx3mx4p[1][1],gx3mx4p[1][0]]]
    gx3x4combined2[2:, 2:] = [[gx3mx4m[1][3],gx3mx4m[1][2]],[gx3mx4m[1][1],gx3mx4m[1][0]]]
    gx3x4n = gx3x4combined2/sum(sum(gx3x4combined2))
    
    corrE = gx3x4n[0,3]-gx3x4n[0,1]+gx3x4n[1,2]-gx3x4n[1,0]+gx3x4n[2,1]-gx3x4n[2,3]+gx3x4n[3,0]-gx3x4n[3,2]

    # if not skipPlot: 
    ticks = np.linspace(-0.5, 2*(nc+0.5), 9)
    ticksL = np.linspace(-0.0, 2*(nc+0.0), 9)
    tick_labelsX = ["","$+p_{B_x}$","$0$","$-p_{B_x}$","","$+p_{B_x}$","$0$","$-p_{B_x}$",""]
    tick_labelsY = ["","$+p_{A_x}$","$0$","$-p_{A_x}$","","$+p_{A_x}$","$0$","$-p_{A_x}$",""]
    # tick_labels = np.concatenate((np.linspace(-4, 4, 5), np.linspace(-4, 4, 5)))
    # plt.imshow(np.flipud(gx3x4combined.T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Greens')
    fig = plt.figure(figsize=figSize)
    fig.subplots_adjust(hspace=0, wspace=sws)
    ax = plt.subplot(1,2,1)
    im = ax.imshow(np.flipud(gx3x4combined.T)*1e6,cmap='Greens',norm=matplotlib.colors.Normalize(vmin=0, vmax=0.25))
    # ax.set_yticks(ticksL, ["","A↗︎","","A↖︎","","A↘︎","","A↙︎",""])
    # ax.set_xticks(ticksL, ["","B↗︎","","B↖︎","","B↘︎","","B↙︎",""])
    ax.set_yticks(ticks, ["","$A$↗︎","","$A$↖︎","","$A$↘︎","","$A$↙︎",""])
    ax.set_xticks(ticks, ["","$B$↙︎","","$B$↘︎","","$B$↖︎","","$B$↗",""])
    
    ax.axhline(y=1.0*nc,color='k',alpha=0.3,linewidth=0.7)
    ax.axhline(y=0.5*nc,color='k',alpha=0.1,linewidth=0.7)
    ax.axhline(y=1.5*nc,color='k',alpha=0.1,linewidth=0.7)
    ax.axvline(x=1.0*nc,color='k',alpha=0.3,linewidth=0.7)
    ax.axvline(x=0.5*nc,color='k',alpha=0.1,linewidth=0.7)
    ax.axvline(x=1.5*nc,color='k',alpha=0.1,linewidth=0.7)
    ax2 = ax.secondary_xaxis('top')
    ax3 = ax.secondary_yaxis('right')
    ax2.set_xticks(ticks, tick_labelsX[::-1])
    ax3.set_yticks(ticks, tick_labelsY)
    plt.title(f"cE={round(corrE,5)}\nt = {t}")
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.50)
    # plt.colorbar(im,pad=0.15,fraction=0.046)
    cbar = plt.colorbar(im,cax=cax)

    # l.info(f"""gx3px4p[1] = {gx3px4p[1]}
    # gx3px4m[1] = {gx3px4m[1]}
    # gx3mx4p[1] = {gx3mx4p[1]}
    # gx3mx4m[1] = {gx3mx4m[1]}""")
    
    ax = plt.subplot(1,2,2)
    im = ax.imshow(np.flipud(gx3x4n.T),cmap='Greens',norm=matplotlib.colors.Normalize(vmin=0, vmax=0.25))
    ticks=np.arange(0,4,1)
    ax.set_yticks(ticks, ["$A$↗︎","$A$↖︎","$A$↘︎","$A$↙︎"])
    ax.set_xticks(ticks, ["$B$↙︎","$B$↘︎","$B$↖︎","$B$↗"])
    ax2 = ax.secondary_xaxis('top')
    ax3 = ax.secondary_yaxis('right')
    # # ax2.set_yticks(ticks, ticks)
    ax3.set_yticks(ticks, ["$+p_{A_x}$","$-p_{A_x}$","$+p_{A_x}$","$-p_{A_x}$"])
    ax2.set_xticks(ticks, ["$+p_{B_x}$","$-p_{B_x}$","$+p_{B_x}$","$-p_{B_x}$"])
    for i in range(gx3x4n.shape[0]):
        for j in range(gx3x4n.shape[1]):
            plt.text(j, i, str(round(np.flipud(gx3x4n.T)[i, j],4)), ha='center', va='center', color='black',fontsize='small')
    plt.title(f"{title2}\ncut = {cutPlot}")
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.50)
    # cbar = plt.colorbar(im,pad=0.15,fraction=0.046)
    # cbar.set_ticks(np.linspace(0,0.25,6))
    cbar = plt.colorbar(im,cax=cax)

    title = f"CorrE t={round(t,5)}, cut = {cutPlot}, s={title2filestr}"
    if saveFig:
        plt.savefig(output_pre_corr+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_corr+title+".png", dpi=600, bbox_inches='tight')
    if skipPlot:
        plt.close()
    else:
        plt.show()
    return (gx3px4p, gx3px4m, gx3mx4p, gx3mx4m, gx3x4n, corrE)





# ### Actually running stuff (🈶)

assert False, "just to catch run all"

# +
plt.figure(figsize=(14,9))

# cut_list = [1, 10, 30]
cut_list = [1, 2, 10]
for i in range(3):
    cut = cut_list[i]
    gx3x4 = gp3p4_dhalo_calc_noAb(phi,cut=cut,offset3=+p,offset4=+p)
    plt.subplot(2,len(cut_list),i+1)
    gx3x4_img = colorize(gx3x4.T)
    
    # xip = pxlin > +0*cut*dpz 
    # xim = pxlin < -0*cut*dpz 
    # gpp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xip],pxlin[xip],axis=0)
    # gpm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xip],pxlin[xip],axis=0)
    # gmp = np.trapz(np.trapz(gx3x4[:,xip],pxlin[xip],axis=1)[xim],pxlin[xim],axis=0)
    # gmm = np.trapz(np.trapz(gx3x4[:,xim],pxlin[xim],axis=1)[xim],pxlin[xim],axis=0)
    # ECHSH = (gpp+gmm-gpm-gmp)/((gpp+gmm+gpm+gmp))
    
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
    # plt.colorbar()
plt.legend(handles=legend_elements, loc='upper right')    

for i in range(3):
    cut = cut_list[i]
    gx3x4, gx3x4config = gp3p4_dhalo_calc(phi,cut=cut,offset3=+p,offset4=+p)
    # gx3x4 = gp3p4_dhalo_calc_noAb(phi,cut=cut,offset3=+p,offset4=+p)
    plt.subplot(2,len(cut_list),i+1+3)
    plot_dhalo_gp3p4(gx3x4,cut)
    # plt.colorbar()

# plt.colorbar()
    
plt.tight_layout(pad=0)
title = "double halo corr with phase"
plt.savefig(output_prefix+title+f" t={round(t,5)}"+".pdf", dpi=600)
plt.savefig(output_prefix+title+f" t={round(t,5)}"+".png", dpi=600)

plt.show()
# -

((np.abs(gx3x4)**2).max())

((np.abs(gx3x4)**2).min())

gx3x4[0].shape

t=1.2052
# t=0.5657
# t=0.6347
# t=0.03
# t = 0.5
# data_folder = "20240512-005555-TFF" 
data_folder = "20240911-182427-TFF"#"20240711-234819-TFF" #"20240528-224811-TFF"
# settingStr = "1-0"
settingStr = "6-5"
psi, phi = None, None
gc.collect()
# with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=8) as file:
with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)} s={settingStr}.pgz.pkl', 'rb', thread=8) as file:
    psi = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=7)
gc.collect()

psi, phi = None, None

psi_phi_plot1(t,-2,psi,phi, plt_show=True, plt_save=True, 
              cmax3x=2e-3,
              cmax4x=2e-3,
              cmax3p=2e-5,
              cmax4p=2e-5,
              xxlima=-xmax,xxlimb=+xmax,xylima=-zmax,xylimb=+zmax,
              pxlima=-pxmax/(hb*k),pxlimb=+pxmax/(hb*k),pzlima=-pzmax/(hb*k),pzlimb=+pzmax/(hb*k),
              xxTmaj=10,xxTmin=5,xyTmaj=10,xyTmin=5,
                pxTmaj=1,pxTmin=0.5,pzTmaj=1,pzTmin=0.5
              )



whatever_bla_testing2 = plot_g34(phi, cutPlot=1.5, saveFig=True, 
        title2=f"Setting: 3-0",
        title2filestr="s3-0", skipPlot=False
        )

t=1.2052
data_folder = "20240919-173152-TFF"
# settingStr = "2-7"
settingStr = selected_sett[0]
psi, phi = None, None
gc.collect()
# with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=8) as file:
with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)} s={settingStr}.pgz.pkl', 'rb', thread=8) as file:
    psi = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=7)
gc.collect()

figSizeHere = (10.5,1.5)
# figSizeHere = (10.5,3.5)
whatever_bla_testing2v2 = plot_g34_v2(phi, cutPlot=1.5, saveFig=True, 
        pMaxCut=2,
        title2=f"s={settingStr}, szh={figSizeHere}",
        title2filestr=f"{settingStr}, fsh={figSizeHere}", skipPlot=False,
        figSize=figSizeHere
        )
del figSizeHere









whatever_bla_testing2v2[4][0,3]-whatever_bla_testing2v2[4][0,1]+\
whatever_bla_testing2v2[4][1,2]-whatever_bla_testing2v2[4][1,0]+\
whatever_bla_testing2v2[4][2,1]-whatever_bla_testing2v2[4][2,3]+\
whatever_bla_testing2v2[4][3,0]-whatever_bla_testing2v2[4][3,2]

whatever_bla_testing2v2[4]

whatever_bla_testing2v2[4][1,2]

(whatever_bla_testing2v2[4][0,3],whatever_bla_testing2v2[4][0,1]
,whatever_bla_testing2v2[4][1,2],whatever_bla_testing2v2[4][1,0]
,whatever_bla_testing2v2[4][2,1],whatever_bla_testing2v2[4][2,3]
,whatever_bla_testing2v2[4][3,0],whatever_bla_testing2v2[4][3,2])

corrE

pxmax/dpx

nx

nc

# +
# 2024 Aug 31 Fig



# -







psi, phi = None, None
gc.collect()





# ### CorrE Visualisation (🈶)

assert False, "just to catch run all"

comboSettSet

# t=0.15
# t=0.2657
# t=0.2957
# t=0.3257
# t=0.3347
# t=0.5657
# t=0.5857
# t=0.6057
# t=0.6257
t=1.2052
# data_folder = "20240521-231755-TFF"
data_folder="20240528-224811-TFF"
(vv3, tt3), (vv4, tt4) = comboSettSet[0][0] 
settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
l.info(f"Loading t={round(t,5)}  s={settingStr}")
# with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)} s={settingStr}.pgz.pkl'
#                 , 'rb', thread=8) as file:
#     psi = pickle.load(file)
# phi, swnf = phiAndSWNF(psi, nthreads=7)
# plot_g34(phi, cutPlot=1.5, saveFig=False, 
#          title2=f"$\phi_3$={round(vv3/VR/0.02)}π/4, $\phi_4=${round(vv4/VR/0.02)}π/4",
#          title2filestr=settingStr
#         )

# assert False, "just to catch run all"


whatever_bla_testing1 = plot_g34(phi, cutPlot=1.5, saveFig=True, 
        title2=f"Initial scattering",
        title2filestr="NA", skipPlot=False
        )

psi_phi_plot1(t,-1,psi,phi, plt_show=True, plt_save=True)





psi=None
phi=None


# find all psi at t=1.2052
def extract_pgzpkl_t(filename):
    # match = re.search(r's=(\d+-\d+)', filename)
    # match = re.search(r's=(\d+-\dD?\d*+)', filename)
    match = re.search(r's=(\d+-\d+(D\d*)?)', filename)
    # return float(match.group(1)) if match else None
    if match: # Remove any non-numeric characters after the number
        # print(match)
        return match.group(1)
    return None


psi_pgzpkl_list = glob.glob(output_prefix+"*t=1.2052*.pgz.pkl")
psi_pgzpkl_list.sort(key=extract_pgzpkl_t)
psi_pgzpkl_ss_l = [extract_pgzpkl_t(x) for x in psi_pgzpkl_list]

psi_pgzpkl_list[20]

extract_pgzpkl_t(psi_pgzpkl_list[20])

psi_pgzpkl_list[:3]

psi_pgzpkl_ss_l[:3]



t=1.2052
outputOfSett = {}
data_folder = "20240919-173152-TFF"#"20240906-124041-TFF" #"20240711-234819-TFF" #"20240528-224811-TFF"
# for combSetH in comboSettSet:
#     for tCombo in combSetH:
# for psi_pgzpkl in psi_pgzpkl_list:
for settingStr in psi_pgzpkl_ss_l:
    # (vv3, tt3), (vv4, tt4) = tCombo
    # settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
    l.info(f"Loading {data_folder} \t t={round(t,5)}  s={settingStr}")
    # del psi, phi, swnf
    # if 'psi' in locals():   del psi
    # if 'phi' in locals():   del phi
    psi, phi = None, None
    # s3 = float(settingStr.split("-")[0])
    # s4 = float(settingStr.split("-")[1])
    if 'D' in settingStr:
        dd = float(settingStr.split("D")[1])
        s3 = float(settingStr.split("D")[0].split("-")[0])
        s4 = float(settingStr.split("D")[0].split("-")[1])
    else:
        s3 = float(settingStr.split("-")[0])
        s4 = float(settingStr.split("-")[1])
        dd = 4 
    thispklfile = f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)} s={settingStr}.pgz.pkl'
    thisPhiCorrFile = output_prefix+f"phi-corr-stats s={settingStr}"+output_ext
    if not os.path.exists(thispklfile): 
        print("AHHHHHHH")
        break
    if os.path.exists(thisPhiCorrFile): 
        with pgzip.open(thisPhiCorrFile,'rb', thread=8) as file:
            tempPlotOutput = pickle.load(file)
            l.info(f"  loaded corrE({settingStr}) = {corrE(np.flipud(tempPlotOutput[4].T))}")
        outputOfSett[settingStr] = tempPlotOutput
        continue

    with pgzip.open(thispklfile, 'rb', thread=8) as file:
        psi = pickle.load(file)
    phi, swnf = phiAndSWNF(psi, nthreads=7)
    
    # (gx3px4p, gx3px4m, gx3mx4p, gx3mx4m, gx3x4n) 
    scattering_evolve_loop_plot(t,settingStr,psi,phi, plt_show=False, plt_save=True)
    gc.collect()
    scattering_evolve_loop_plot_alt(t,settingStr,psi,phi, plt_show=False, plt_save=True, logPlus=1, po=0.1)
    gc.collect()
    psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",
                    cmax3x=2e-3,cmax4x=2e-3,cmax3p=2e-5,cmax4p=2e-5)
    # tempPlotOutput = plot_g34(
    gc.collect()
    psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",
                    cmax3x=5e-3,cmax4x=5e-3,cmax3p=5e-5,cmax4p=5e-5)
    gc.collect()
    psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True,save_str=f"s={settingStr}",title_str=f"s={settingStr}",
                    cmax3x=1e-3,cmax4x=1e-3,cmax3p=1e-5,cmax4p=1e-5)
    gc.collect()
    tempPlotOutput = plot_g34_v2(
        phi, cutPlot=1.5, saveFig=True, 
        pMaxCut=2,
        title2=f"$\phi_3$={s3}π/4, $\phi_4=${s4}π/4",
        title2filestr=settingStr, skipPlot=True
        )
    # l.info(tempPlotOutput)
    with pgzip.open(thisPhiCorrFile,'wb', thread=8, blocksize=1*10**8) as file:
        pickle.dump(tempPlotOutput, file)

    outputOfSett[settingStr] = tempPlotOutput
    # del psi, phi, swnf
    l.info(f"  calced corrE({settingStr}) = {corrE(np.flipud(tempPlotOutput[4].T))}")
    gc.collect()
with pgzip.open(output_prefix+f"outputOfSett"+output_ext,'wb', thread=8, blocksize=1*10**8) as file:
    pickle.dump(outputOfSett, file)

with pgzip.open("/Volumes/tonyNVME Gold/twoParticleSim/20240711-234819-TFF/outputOfSett.pgz.pkl", 'rb', thread=1) as file:
    outputOfSett = pickle.load(file)

phi_corrE_list = glob.glob(output_prefix+"phi-corr-stats*")
phi_corrE_list.sort(key=extract_pgzpkl_t)
phi_corrE_ss_l = [extract_pgzpkl_t(x) for x in phi_corrE_list]

# +
selected_sett = ["2-7","2-1","0-7","0-1"]
corrE_3p4 = []
corrE_3m4 = []
corrE_3a4 = [] 
corrE_val = []
corrE_3p4s = []
corrE_3m4s = []
corrE_3a4s = []
corrE_vals = []  
for (pcI, pcE) in enumerate(phi_corrE_list):
    sss = extract_pgzpkl_t(pcE)
    if 'D' in sss:
        dd = float(sss.split("D")[1])
        s3 = float(sss.split("D")[0].split("-")[0])
        s4 = float(sss.split("D")[0].split("-")[1])
    else:
        s3 = float(sss.split("-")[0])
        s4 = float(sss.split("-")[1])
        dd = 4 

    # print(sss,s3,s4, dd)
    # continue
    with pgzip.open(pcE,'rb', thread=2) as file:
        tempPlotOutput = pickle.load(file)
    cE = corrE(np.flipud(tempPlotOutput[4].T))
    print(f"corrE({sss}) = {cE}")

    if sss in selected_sett:
        corrE_3p4s.append((s3+s4)*pi/dd)
        corrE_vals.append(cE)
        corrE_3m4s.append((s3-s4)*pi/dd)
        corrE_3a4s.append((s3,s4))
    else:
        corrE_3p4.append((s3+s4)*pi/dd)
        corrE_val.append(cE)
        corrE_3m4.append((s3-s4)*pi/dd)
        corrE_3a4.append((s3,s4))


corrE_3p4 = np.array(corrE_3p4)
corrE_3m4 = np.array(corrE_3m4)
corrE_3a4 = np.array(corrE_3a4)
corrE_val = np.array(corrE_val)
corrE_3p4s = np.array(corrE_3p4s)
corrE_vals = np.array(corrE_vals)
corrE_3m4s = np.array(corrE_3m4s)
corrE_3a4s = np.array(corrE_3a4s)

# -

float(sss.split("-")[0])

corrE_3p4s


# +
def cosM(x, a, o, h):
    return a*np.cos(x-o)+h
par, cov = scipy.optimize.curve_fit(cosM, corrE_3p4, corrE_val)
par_sig = np.sqrt(np.diag(cov))
xxx = np.linspace(min(corrE_3p4)-1,max(corrE_3p4)+1,1000)
xxe = np.linspace(min(corrE_3p4)-10,max(corrE_3p4)+10,3000)
yyy = cosM(xxx, *par)
yys = 2 * par_sig[0] * np.cos(xxx-par[1])
yyd = np.sqrt(np.sum(par_sig**2))

plt.figure(figsize=(11,4))
# plt.fill_between(xxe, -1, -1/sqrt(2), color='gray', alpha=0.2, label="_nolegend_",linewidth=0)
# plt.fill_between(xxe, +1/sqrt(2), +1, color='gray', alpha=0.2, label="_nolegend_",linewidth=0)
plt.fill_between(xxe, -1, -0.5, color='gray', alpha=0.2, label="_nolegend_")
plt.fill_between(xxe, +0.5, +1, color='gray', alpha=0.2, label="_nolegend_")

plt.fill_between(xxx, yyy-2*yys, yyy+2*yys, color='purple', alpha=0.6, label="±2σ Fit")
plt.scatter(corrE_3p4,corrE_val,30,marker='x',alpha=0.8, label="Simulation")
plt.scatter(corrE_3p4s,corrE_vals,30,marker='x',alpha=0.9, color="lime", label="_nolegend_")
l.info(f"a = {par[0]} ± {par_sig[0]}")
l.info(f"o = {par[1]} ± {par_sig[1]}")
l.info(f"h = {par[2]} ± {par_sig[2]}")
plt.xlabel("$(\\varphi_{A} + \\varphi_B)/\\pi$")
plt.ylabel("$E(\\varphi_{ A},\\varphi_{ B})$")
plt.yticks(np.linspace(-1,1,4+1))
# plt.xticks(np.linspace(0,2*2*pi,8+1), [str(int(x*2))+"π/4" for x in np.linspace(0,8,8+1)] )
# plt.xticks(np.linspace(0,2*2*pi,8+1), ["0","$\\pi/2$","$\\pi$","$3\\pi/2$","$2\\pi$","${5\\pi}/{2}$","$3\\pi$","${7\\pi}/{2}$","$4\\pi$"])
# plt.xticks(np.linspace(0,2*2*pi,16+1), ["0","$\\pi/4$", "$\\pi/2$", "$3\\pi/4$", "$\\pi$", "$5\\pi/4$", "$3\\pi/2$", "$7\\pi/4$", "$2\\pi$", "$9\\pi/4$", "$5\\pi/2$", "$11\\pi/4$", "$3\\pi$", "$13\\pi/4$", "$7\\pi/2$", "$15\\pi/4$", "$4\\pi$"])
plt.xticks(np.linspace(0,2*2*pi,16+1),np.linspace(0,4,16+1))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pi/8))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
plt.xlim(0-0.5,2*2*pi+0.5)
plt.ylim(-1,+1)
plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
# plt.text(4*pi-0.8,-0.81,"Non-local")
# plt.text(4*pi-0.8,-0.67,"Local")
plt.text(4*pi-0.8,-0.61,"Non-local")
plt.text(4*pi-0.8,-0.46,"Local")
# plt.text(4*pi-0.7,+0.38,"Non-local")
# plt.text(4*pi-0.7,+0.55,"Local")
plt.legend(loc='lower right')

plt.savefig(output_prefix+"corrE-3p4-v3b.pdf", dpi=600, bbox_inches='tight')
plt.savefig(output_prefix+"corrE-3p4-v3b.png", dpi=600, bbox_inches='tight')
plt.show()
# -

["0","$\\frac{\\pi}{2}$","$\\pi$","$\\frac{3\\pi}{2}$","$2\\pi$","$\\frac{5\\pi}{2}$","$3\\pi$","$\\frac{7\\pi}{2}$","$4\\pi$"] 

plt.scatter(corrE_3m4, corrE_val)







outputOfSett["0-1"][0][0].shape

outputOfSett["0-1"][4]

outputOfSett["0-1"][4].T

np.flipud(outputOfSett["0-1"][4].T)

gx3px4p


def corrE(g):
    return (g[0,2]+g[2,0]-g[0,0]-g[2,2])/(g[0,2]+g[2,0]+g[0,0]+g[2,2])


corrE(np.flipud(outputOfSett["0-1"][4].T))

1/sqrt(2)

 corrE(np.flipud(outputOfSett["0-1"][4].T))\
+corrE(np.flipud(outputOfSett["6-3"][4].T))\
+corrE(np.flipud(outputOfSett["6-1"][4].T))\
-corrE(np.flipud(outputOfSett["0-3"][4].T))

 corrE(np.flipud(outputOfSett["3-6"][4].T))\
+corrE(np.flipud(outputOfSett["1-0"][4].T))\
+corrE(np.flipud(outputOfSett["1-6"][4].T))\
-corrE(np.flipud(outputOfSett["3-0"][4].T))

 corrE(np.flipud(outputOfSett["5-4"][4].T))\
+corrE(np.flipud(outputOfSett["3-6"][4].T))\
+corrE(np.flipud(outputOfSett["3-4"][4].T))\
-corrE(np.flipud(outputOfSett["5-6"][4].T))

 corrE(np.flipud(outputOfSett["6-3"][4].T))\
+corrE(np.flipud(outputOfSett["4-5"][4].T))\
+corrE(np.flipud(outputOfSett["4-3"][4].T))\
-corrE(np.flipud(outputOfSett["6-5"][4].T))

2*sqrt(2)

corrE(np.flipud(outputOfSett["6-1"][4].T))

corrE(np.flipud(outputOfSett["6-3"][4].T))

corrE(np.flipud(outputOfSett["0-3"][4].T))

np.flipud(outputOfSett["0-3"][4].T)

(corrE(np.flipud(outputOfSett["0-1"][4].T))\
,corrE(np.flipud(outputOfSett["0-3"][4].T))\
,corrE(np.flipud(outputOfSett["1-0"][4].T))\
,corrE(np.flipud(outputOfSett["0-1"][4].T)))





cEResults = {}
for (ci, combSetH) in enumerate(comboSettSet):
    (tC1, tC2, tC3, tC4) = combSetH
    (v31,_), (v41,_) = tC1 
    (v32,_), (v42,_) = tC2
    (v33,_), (v43,_) = tC3
    (v34,_), (v44,_) = tC4
    ss1 = f"{round(v31/VR/0.02)}-{round(v41/VR/0.015)}"
    ss2 = f"{round(v32/VR/0.02)}-{round(v42/VR/0.015)}"
    ss3 = f"{round(v33/VR/0.02)}-{round(v43/VR/0.015)}"
    ss4 = f"{round(v34/VR/0.02)}-{round(v44/VR/0.015)}"
        # (vv3, tt3), (vv4, tt4) = tCombo
        # settingStr = f"{round(vv3/VR/0.02)}-{round(vv4/VR/0.015)}"
        # print(settingStr)
    cE = corrE(np.flipud(outputOfSett[ss1][4].T))\
        -corrE(np.flipud(outputOfSett[ss2][4].T))\
        +corrE(np.flipud(outputOfSett[ss3][4].T))\
        +corrE(np.flipud(outputOfSett[ss4][4].T))
    
    print(f"ci={ci}, g({ss1})-g({ss2})+g({ss3})+g({ss4}) \t cE = {round(cE,5)}")
    print(f"    {round(corrE(np.flipud(outputOfSett[ss1][4].T)),4)}"+\
          f"    {round(corrE(np.flipud(outputOfSett[ss2][4].T)),4)}"+\
          f"    {round(corrE(np.flipud(outputOfSett[ss3][4].T)),4)}"+\
          f"    {round(corrE(np.flipud(outputOfSett[ss4][4].T)),4)}")

comboSettSet[5]

comboSettSet[5][0]

4005/VR/0.02

comboSettingsGen(thetaCombo5)

gc.collect()



# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### CorrE alt - not good?
# -

assert False, "just to catch run all"

1/sqrt(2)

sideP = sqrt((3*m3-m4)/(m3+m4))*hb*k
l.info(f"sideP = {sideP}")

# +
cutPlot=1
gx3px4p = gp3p4_dhaloUD_calc(phi,cut=cutPlot,offset3=+sideP,offset4=+sideP)
gx3px4m = gp3p4_dhaloUD_calc(phi,cut=cutPlot,offset3=+sideP,offset4=-sideP)
gx3mx4p = gp3p4_dhaloUD_calc(phi,cut=cutPlot,offset3=-sideP,offset4=+sideP)
gx3mx4m = gp3p4_dhaloUD_calc(phi,cut=cutPlot,offset3=-sideP,offset4=-sideP)
gx3x4combined = np.zeros((2*nx,2*nx))
gx3x4combined[:nx, :nx] = gx3px4p[0]
gx3x4combined[:nx, nx:] = gx3px4m[0]
gx3x4combined[nx:, :nx] = gx3mx4p[0]
gx3x4combined[nx:, nx:] = gx3mx4m[0]

plt.figure(figsize=(11,5))
ax = plt.subplot(1,2,1)
ax.imshow(np.flipud(gx3x4combined.T),cmap="Greens")
ticks = np.linspace(0, 2*nx, 17)
ticksL = np.linspace(0, 2*nx, 9)
tick_labels = ["","+3","+2","+1","0","-1","-2","-3","","+3","+2","+1","0","-1","-2","-3",""]
ax.set_yticks(ticks, ["","","","A↖︎","","A↙︎","","","","","","A↗︎","","A↘︎","","",""]) # ↗︎↘︎↖︎↙︎
ax.set_xticks(ticks, ["","","","B↘︎","","B↗︎","","","","","","B↙︎","","B↖︎","","",""]) # ↗︎↘︎↖︎↙︎

ax.axhline(y=1.0*nx,color='k',alpha=0.3,linewidth=0.7)
ax.axhline(y=0.5*nx,color='k',alpha=0.1,linewidth=0.7)
ax.axhline(y=1.5*nx,color='k',alpha=0.1,linewidth=0.7)
ax.axvline(x=1.0*nx,color='k',alpha=0.3,linewidth=0.7)
ax.axvline(x=0.5*nx,color='k',alpha=0.1,linewidth=0.7)
ax.axvline(x=1.5*nx,color='k',alpha=0.1,linewidth=0.7)
ax2 = ax.secondary_xaxis('top')
ax3 = ax.secondary_yaxis('right')
ax2.set_xticks(ticks, tick_labels[::-1])
ax3.set_yticks(ticks, tick_labels)
plt.title(f"t = {t}")

l.info(f"""gx3px4p[1] = {gx3px4p[1]}
gx3px4m[1] = {gx3px4m[1]}
gx3mx4p[1] = {gx3mx4p[1]}
gx3mx4m[1] = {gx3mx4m[1]}""")

gx3x4combined = np.zeros((2*2,2*2))
gx3x4combined[:2, :2] = [[gx3px4p[1][3],gx3px4p[1][2]],[gx3px4p[1][1],gx3px4p[1][0]]]
gx3x4combined[:2, 2:] = [[gx3px4m[1][3],gx3px4m[1][2]],[gx3px4m[1][1],gx3px4m[1][0]]]
gx3x4combined[2:, :2] = [[gx3mx4p[1][3],gx3mx4p[1][2]],[gx3mx4p[1][1],gx3mx4p[1][0]]]
gx3x4combined[2:, 2:] = [[gx3mx4m[1][3],gx3mx4m[1][2]],[gx3mx4m[1][1],gx3mx4m[1][0]]]
gx3x4n = gx3x4combined/sum(sum(gx3x4combined))
ax = plt.subplot(1,2,2)
im = ax.imshow(np.flipud(gx3x4combined.T),cmap='Greens')
ticks=np.arange(0,4,1)
ax.set_yticks(ticks, ["A↖︎","A↙︎","A↗︎","A↘︎"])
ax.set_xticks(ticks, ["B↘︎","B↗︎","B↙︎","B↖︎"])
# ax2 = ax.secondary_xaxis('top')
# ax3 = ax.secondary_yaxis('right')
# ax2.set_yticks(ticks, ["A↗︎","A↖︎","A↘︎","A↙︎"])
# ax3.set_xticks(ticks, ["B↙︎","B↘︎","B↖︎","B↗"])
for i in range(gx3x4n.shape[0]):
    for j in range(gx3x4n.shape[1]):
        plt.text(j, i, str(round(np.flipud(gx3x4n.T)[i, j],4)), ha='center', va='center', color='dodgerblue')
plt.title(f"cut = {cutPlot}")
title = f"CorrE alt t={round(t,5)}, cut = {cutPlot}"
plt.savefig(output_prefix+title+".pdf", dpi=600)
plt.savefig(output_prefix+title+".png", dpi=600)
plt.show()
# -













scattering_evolve_loop_plot(t,f,psi,phi, plt_show=True, plt_save=False)

scattering_evolve_loop_plot_alt(t,f,psi,phi, plt_show=True, plt_save=False, logPlus=1, po=0.1)



































# # Exporting to Video (Quite high VM RAM usage)

assert False, "just to catch run all"


# Function to extract frame number from filename
def extract_frame_number(filename):
    match = re.search(r't=([0-9.])', filename)
    # return float(match.group(1)) if match else None
    if match: # Remove any non-numeric characters after the number
        number = match.group(1).rstrip('.')
        return float(number)
    return None
# img_alt_list = glob.glob(output_pre_selp+'*.png')
img_alt_list = glob.glob(output_pre_ppp+'*.png')
# img_alt_list = glob.glob(output_pre_selpa+'*.png')
# img_alt_list = glob.glob(output_prefix+'BS_L free time scan ok 3/*.png')
img_alt_list.sort(key=extract_frame_number)

extract_frame_number(img_alt_list[2])

img_alt_list[:30]

re.search(r't=([0-9.])', img_alt_list[2])

re.search(r't=([0-9.]).*s=sc', img_alt_list[2])

N_JOBS

pattern = r"f=(\d+),.*s=sc\.png"
directory = output_pre_ppp
img_alt_list = sorted(
    #  (int(re.match(pattern, f).group(1)), f) for f in os.listdir(output_pre_ppp) if re.match(pattern, f)
    # os.path.join(directory, f) for f in os.listdir(directory) if re.match(pattern, f)
    [os.path.join(directory, f) for f in os.listdir(directory) if (m := re.match(pattern, f))],
    key=lambda x: int(re.match(pattern, os.path.basename(x)).group(1))
)

img_alt_list

# img_alt_frames = []  # Read and process images, storing them in a list
# for image in tqdm(img_alt_list, desc="Processing Images"):
#     img = cv2.imread(image)
#     img_alt_frames.append(img)  # Append processed frame to list
img_alt_frames = Parallel(n_jobs=-1)(
    delayed(lambda image: cv2.imread(image))(image) for image in tqdm(img_alt_list, desc="Processing Images")
)

if img_alt_frames:  # Determine the width and height from the first image if not empty
    height, width, layers = img_alt_frames[0].shape # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'hvc1')  # HEVC codec
    out = cv2.VideoWriter(
        # output_prefix+"scattering_evolve_loop_plot.mov", 
        output_prefix+"init-scattering.mov",
        # output_prefix+"scattering_evolve_loop_plot_alt.mov", 
        # output_prefix+"BS_L free time scan ok 3.mov",
        fourcc, 10.0, (width, height), True) # Write img_alt_frames to the video file
    for frame in tqdm(img_alt_frames, desc="Writing Video"):
        out.write(frame)  # Write out frame to video
    out.release()  # Release the video writer
    del frame, out
else:
    print("No images found or processed.")
del img_alt_frames
gc.collect()

cv2.cuda.printCudaDeviceInfo(0)

cv2.cuda.getCudaEnabledDeviceCount()





# + active=""
# # Create a clip from the images sequence
# clip = ImageSequenceClip(img_alt_list, fps=10)  # fps can be adjusted
# # Write the clip to a video file in MOV format
# clip.write_videofile(output_prefix+"scattering_evolve_loop_plot_alt.mp4",threads=8,fps=10.0)
# # Close the clip to free resources
# clip.close()

# + active=""
# # Determine the width and height from the first image
# frame = cv2.imread(img_alt_list[0])
# height, width, layers = frame.shape
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'hvc1')  # Be sure to use lower case
# out = cv2.VideoWriter(output_prefix+"scattering_evolve_loop_plot_alt.mov", fourcc, 10.0, (width, height))
#
# for image in tqdm(img_alt_list):
#     img = cv2.imread(image)
#     out.write(img)  # Write out frame to video
#
# out.release()  # Release the video writer
# -

gc.collect()
# %reset -f in
# %reset -f out
ram_py_log()
print_ram_usage(globals().items(),10)

















# # Making Figures

# ### Function Defs (🔄)

output_pre_ppp = output_prefix + "psi_phi_plot/"
os.makedirs(output_pre_ppp, exist_ok=True)
def psi_phi_plot1(t,f,psi,phi, 
                  plt_show=True, plt_save=False, 
                  save_str="", title_str="",
                  cmax3x=1,cmax3p=1,cmax4x=1,cmax4p=1,
                  xxlima=-xmax,xxlimb=+xmax,xylima=-zmax,xylimb=+zmax,
                  pxlima=-pxmax/(hb*k),pxlimb=+pxmax/(hb*k),pzlima=-pzmax/(hb*k),pzlimb=+pzmax/(hb*k),
                  xxTmaj=10,xxTmin=5,xyTmaj=10,xyTmin=5,
                  pxTmaj=1,pxTmin=0.5,pzTmaj=1,pzTmin=0.5,
                  cbar=False
                  ):
    t_str = str(round(t,5))
    if plt_show:
        print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
              str(round(ram_py_MB(),3)) + "MB  ")
    # normC = matplotlib.colors.Normalize(vmin=0, vmax=cmax)
    # normP = matplotlib.colors.Normalize(vmin=0, vmax=cmaxp)
    fig = plt.figure(figsize=(8,8.5))
    plt.subplot(2,2,1)
    plt.imshow(np.flipud(only3(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax3x))
    plt.xlabel("$x \ (\mu m)$", labelpad=0)
    plt.ylabel("$z \ (\mu m)$", labelpad=-10)
    plt.title("He$^3\ \psi$")
    plt.xlim(xxlima,xxlimb)
    plt.ylim(xylima,xylimb)
    # plt.title("$t="+t_str+" \ ms $")
    # add a label to the top left corner
    # plt.text(-xmax*0.95,zmax*0.95,"He$^3\ \psi$",color='k',ha='left',va='top',alpha=0.9)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=xxTmaj))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=xxTmin))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=xyTmaj))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=xyTmin))

    plt.subplot(2,2,2)
    plt.imshow(np.flipud(only4(psi).T), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax4x))
    plt.xlabel("$x \ (\mu m)$", labelpad=0)
    plt.ylabel("$z \ (\mu m)$", labelpad=-10)
    plt.title("He$^4\ \psi$")
    plt.xlim(xxlima,xxlimb)
    plt.ylim(xylima,xylimb)
    # plt.text(-xmax*0.95,zmax*0.95,"He$^4\ \psi$",color='k',ha='left',va='top',alpha=0.9)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=xxTmaj))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=xxTmin))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=xyTmaj))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=xyTmin))

    plt.subplot(2,2,3)
    plt.imshow((only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax3p))
    plt.xlabel("$p_x \ (\hbar k)$", labelpad=0)
    plt.ylabel("$p_z \ (\hbar k)$", labelpad=-10)
    plt.title("He$^3\ \phi$")
    plt.xlim(pxlima,pxlimb)
    plt.ylim(pzlima,pzlimb)
    # plt.text(-pxmax*0.95/p,pzmax*0.95/p,"He$^3\ \phi$",color='k',ha='left',va='top',alpha=0.9)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=pxTmaj))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=pxTmin))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=pzTmaj))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=pzTmin))
    # add color bar
    if cbar: plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow((only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax4p))
    plt.xlabel("$p_x \ (\hbar k)$", labelpad=0)
    plt.ylabel("$p_z \ (\hbar k)$", labelpad=-10)
    plt.title("He$^34\ \phi$")
    plt.xlim(pxlima,pxlimb)
    plt.ylim(pzlima,pzlimb)
    # plt.text(-pxmax*0.95/p,pzmax*0.95/p,"He$^4\ \phi$",color='k',ha='left',va='top',alpha=0.9)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=pxTmaj))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=pxTmin))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=pzTmaj))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=pzTmin))
    if cbar: plt.colorbar()

    # add suptitle
    if title_str=="": plt.suptitle("t = "+t_str+" ms")
    else: plt.suptitle("t = "+t_str+" ms, "+title_str)

    if plt_save:
        title= "f="+str(f)+",t="+t_str+","+str((cmax3x,cmax3p,cmax4x,cmax4p))+save_str
        plt.savefig(output_pre_ppp+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_ppp+title+".png", dpi=600, bbox_inches='tight')
    
    if plt_show: plt.show() 
    else:        plt.close(fig) 
    gc.collect()


def psi_phi_plot2(t,f,psi,phi, titleR="",
                  o3xM=0.050, o4xM=0.050, o3pM=0.00050, o4pM=0.00050,
                  lP=1, po=0.4, mX3=1, mX4=1, mP3=1, mP4=1, bX=1e2, bP=1e4,
                  labPosX=-1, labPosY=-7,
                  momXPlotWid=pxmax/(hb*k), momZPlotWid=pzmax/(hb*k),
                  skipPlot=False, saveFig=True,
                  settingStr="NA"
                  ):
    t_str = str(round(t,5))
    # print(f"t={t_str}, f={f}, RMA:{round(ram_py_MB(),3)}MB")
    titleL = f"t={t_str}ms, f={f}"

    fig = plt.figure(figsize=(7,6))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    # 1st Row Position
    cb3tlins = np.linspace(0,o3xM,20+1)
    cb3ticks = mX3*np.power(np.log(lP+bX*cb3tlins),po)
    cb3tlbls = ['0.00', '0.05', '0.10', '0.15', '0.20', '', '0.30', '', '0.40', '', '0.50', '', '', '', '0.70', '', '', '', '0.90', '','']
    cb4tlins = np.linspace(0,o4xM,20+1)
    cb4ticks = mX4*np.power(np.log(lP+bX*cb4tlins),po)
    cb4tlbls = cb3tlbls

    ax = plt.subplot(2,2,1)
    im = ax.imshow(mX3*np.power(np.log(lP+bX*only3(psi).T),po), extent=[-xmax,xmax,-zmax,zmax],cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb3ticks)))
    ax.set_xlabel("$x (\mu m)$", labelpad=labPosX, fontsize=10)
    ax.set_ylabel("$z (\mu m)$", labelpad=labPosY, fontsize=10)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(cb3ticks); cbar.set_ticklabels(cb3tlbls); cbar.ax.tick_params(labelsize=8)
    ax.set_title(titleL+"\n${}^3\\text{He}^\\ast$",fontsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))

    ax = plt.subplot(2,2,2)
    im = ax.imshow(mX4*np.power(np.log(lP+bX*only4(psi).T),po), extent=[-xmax,xmax,-zmax,zmax],cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb4ticks)))
    ax.set_xlabel("$x (\mu m)$", labelpad=labPosX, fontsize=10)
    # ax.set_ylabel("$z (\mu m)$", labelpad=labPosY, fontsize=10)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(cb4ticks); cbar.set_ticklabels(cb4tlbls); cbar.ax.tick_params(labelsize=8)
    ax.set_title(titleR+"\n${}^4\\text{He}^\\ast$",fontsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=5))

    # 2nd Row Momentum 
    cb3tlins = np.linspace(0,o3pM,20+1)
    cb3ticks = mP3*np.power(np.log(lP+bP*cb3tlins),po)
    # cb3tlbls = [format(x,'.2f') if (i%2==0) else "" for (i, x) in enumerate(cb3tlins*0.05/(cb3tlins[1]-cb3tlins[0]))]
    # cb3tlbls = ['0.00', '0.05', '0.10', '0.15', '0.20', '', '0.30', '', '0.40', '', '0.50', '', '', '', '0.70', '', '', '', '0.90', '','']
    # cb3tlbls = ['0.00', '0.05', '0.10', '0.15', '0.20', '', '0.30', '', '0.40', '', '0.50', '', '0.60', '', '', '', '0.80', '', '', '','']
    cb4tlins = np.linspace(0,o4pM,20+1)
    cb4ticks = mP4*np.power(np.log(lP+bP*cb4tlins),po)
    # cb4tlbls = [format(x,'.2f') if i<7 else "" for (i, x) in enumerate(cb4tlins*0.05/(cb4tlins[1]-cb4tlins[0]))]
    # cb4tlbls = cb3tlbls

    ax = plt.subplot(2,2,3)
    im = ax.imshow(mP3*np.power(np.log(lP+bP*only3phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb3ticks)))
    ax.set_xlabel("$p_x \ (\hbar k)$", labelpad=labPosX, fontsize=10)
    ax.set_ylabel("$p_z \ (\hbar k)$", labelpad=labPosY, fontsize=10)
    ax.set_xlim(-momXPlotWid,+momXPlotWid)
    ax.set_ylim(-momZPlotWid,+momZPlotWid)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(cb3ticks); cbar.set_ticklabels(cb3tlbls); cbar.ax.tick_params(labelsize=8)
    # plt.title("${}^3\\text{He}^\\ast$")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.5))

    ax = plt.subplot(2,2,4)
    im = ax.imshow(mP4*np.power(np.log(lP+bP*only4phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb4ticks)))
    ax.set_xlabel("$p_x \ (\hbar k)$", labelpad=labPosX, fontsize=10)
    # plt.ylabel("$p_z \ (\hbar k)$", labelpad=labPosY, fontsize=10)
    ax.set_xlim(-momXPlotWid,+momXPlotWid)
    ax.set_ylim(-momZPlotWid,+momZPlotWid)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(cb4ticks); cbar.set_ticklabels(cb4tlbls); cbar.ax.tick_params(labelsize=8)
    # plt.title("${}^4\\text{He}^\\ast$")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.5))

    title= "f="+str(f)+",t="+t_str+",lP="+str(lP)+f" s={settingStr}"
    if saveFig: 
        plt.savefig(output_pre_ppp+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_pre_ppp+title+".png", dpi=600, bbox_inches='tight')
    if skipPlot: plt.close(fig)
    else: plt.show()
gc.collect()



# ### Plottings (🈶)



psi_phi_plot1(t,-3,psi,phi, plt_show=True, plt_save=True, 
            cmax3x=2e-3,
            cmax4x=2e-3,
            cmax3p=3e-5,
            cmax4p=3e-5,
            xxlima = -xmax,
            xxlimb = +xmax,
            xylima = -zmax,
            xylimb = +zmax,
            pxlima = -1.5,
            pxlimb = +1.5,
            pzlima = -2.5,
            pzlimb = +2.5,
            xxTmaj = 10,
            xxTmin = 5,
            xyTmaj = 10,
            xyTmin = 5,
            pxTmaj = 1,
            pxTmin = 0.25,
            pzTmaj = 1,
            pzTmin = 0.25,
            cbar=True
              )

scattering_evolve_loop_plot_alt(t,-2,psi,phi, plt_show=True, plt_save=False, logPlus=1, po=0.1)

1e-20



output_prefix

# t=1.2052
# t=0.5657
# t=0.6347
t = 0.03
data_folder = "20241000-000000-TFF"#"20240919-173152-TFF"
# data_folder = "20240512-005555-TFF" 
# data_folder = "20240911-182427-TFF"#"20240711-234819-TFF" #"20240528-224811-TFF"
settingStr = "1-2"
# settingStr = "6-5"
psi, phi = None, None
gc.collect()
with pgzip.open(f'output/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=16) as file:
# with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)} s={settingStr}.pgz.pkl', 'rb', thread=8) as file:
    psi = pickle.load(file)
phi, swnf = phiAndSWNF(psi, nthreads=7)
gc.collect()


def fucking_scaling_fuck(x, cutoff, amp, bb=1):
    # if x <= cutoff: return cutoff
    # else: return np.arctan(x-cutoff)/(pi/2)*amp + cutoff
    return np.where(x <= cutoff, x, np.arctan(bb*(x - cutoff) / (np.pi / 2) * amp + cutoff))


# +
logPlus=1; po=0.4; f=-7
# o3pM = np.max(only3phi(phi))
# o4pM = np.max(only4phi(phi))
print(np.max(only3phi(phi)),np.max(only4phi(phi)))
# o3pM = 0.0003863759165270296
# o4pM = 0.00016002514388641225
o3pM = 0.00032
o4pM = 0.00014
mmm = 1e-6
mm3 = 1
mm4 = 1
lm = 1e5
atm=6e4
cmax3p= 5e-5
cmax4p= 5e-5
camp3 = 0e-5
camp4 = 0e-5
bbb = 1e5
bbc = 1e4

cb3tlins = np.linspace(0,o3pM,20)
cb3ticks = mm3*np.power(np.log(logPlus+bbc*cb3tlins),po)
cb3tlbls = [format(x,'.2f') if i<7 else "" for (i, x) in enumerate(cb3tlins*0.05/(cb3tlins[1]-cb3tlins[0]))]
cb4tlins = np.linspace(0,o4pM,20)
cb4ticks = mm4*np.power(np.log(logPlus+bbc*cb4tlins),po)
cb4tlbls = [format(x,'.2f') if i<7 else "" for (i, x) in enumerate(cb4tlins*0.05/(cb4tlins[1]-cb4tlins[0]))]

t_str = str(round(t,5))
print("t = " +t_str+ " \t\t frame =", f, "\t\t memory used: " + 
        str(round(ram_py_MB(),3)) + "MB  ")

fig = plt.figure(figsize=(6,3))
# plt.subplot(2,2,1)
# plt.imshow(np.power(np.log(logPlus+np.flipud(only3(psi).T)),po), extent=[-xmax,xmax,-zmax, zmax],cmap='Reds')
# plt.xlabel("$x \ (\mu m)$")
# plt.ylabel("$z \ (\mu m)$")
# plt.title("$t="+t_str+" \ ms $")

# plt.subplot(2,2,2)
# plt.imshow(np.power(np.log(logPlus+np.flipud(only4(psi).T)),po), extent=[-xmax,xmax,-zmax, zmax],cmap='Blues')
# plt.xlabel("$x \ (\mu m)$")
# plt.ylabel("$z \ (\mu m)$")

plt.subplot(1,2,1)
# plt.imshow(fucking_scaling_fuck(only3phi(phi).T,cmax3p,camp3,bbb), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
# plt.imshow(np.power(np.log(1+lm*only3phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax3p))
# plt.imshow(fucking_scaling_fuck(np.power(np.log(1+lm*only3phi(phi).T),po),cmax3p,camp3), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
# plt.imshow(np.arctan(atm*only3phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds')
plt.imshow(mm3*np.power(np.log(logPlus+bbc*only3phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb3ticks)))
# plt.imshow(np.power(np.log(logPlus+only3phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Reds',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax3p))
plt.xlabel("$p_x \ (\hbar k)$")
plt.ylabel("$p_z \ (\hbar k)$")
plt.xlim(-1.5,+1.5)
plt.ylim(-2.5,+2.5)
cbar = plt.colorbar()
cbar.set_ticks(cb3ticks)
cbar.set_ticklabels(cb3tlbls)
cbar.ax.tick_params(labelsize=8)
plt.title("${}^3\\text{He}^\\ast$")

plt.subplot(1,2,2)
# plt.imshow(fucking_scaling_fuck(only4phi(phi).T,cmax4p,camp4,bbb), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
# plt.imshow(np.power(np.log(1+lm*only4phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax4p))
# plt.imshow(fucking_scaling_fuck(np.power(np.log(1+lm*only4phi(phi).T),po),cmax4p,camp4), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
# plt.imshow(np.arctan(atm*only4phi(phi).T), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues')
plt.imshow(mm4*np.power(np.log(logPlus+bbc*only4phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(cb4ticks)))
# plt.imshow(np.power(np.log(logPlus+only4phi(phi).T),po), extent=np.array([-pxmax,pxmax,-pzmax,pzmax])/(hb*k),cmap='Blues',norm=matplotlib.colors.Normalize(vmin=0, vmax=cmax4p))
plt.xlabel("$p_x \ (\hbar k)$")
plt.xlabel("$p_x \ (\hbar k)$")
plt.xlim(-1.5,+1.5)
plt.ylim(-2.5,+2.5)
cbar = plt.colorbar()
cbar.set_ticks(cb4ticks)
cbar.set_ticklabels(cb4tlbls)
cbar.ax.tick_params(labelsize=8)
plt.title("${}^4\\text{He}^\\ast$")

title= "f="+str(f)+",t="+t_str+",logPlus="+str(logPlus)+f" s={settingStr}"
plt.savefig(output_pre_selpa+title+".pdf", dpi=600, bbox_inches='tight')
plt.savefig(output_pre_selpa+title+".png", dpi=600, bbox_inches='tight')
plt.show()
gc.collect()
# -



psi_phi_plot2(t=t,f=-8,psi=psi,phi=phi)

_ = plot_g34_v2(t=t, phiHere=phi)



xmax

print(cb3tlbls)

print([format(x,'.2f') if (True) else "" for (i, x) in enumerate(cb3tlins*0.05/(cb3tlins[1]-cb3tlins[0]))])

0.05 * np.size(cb3tlins)

5/v3

o3pM

o4pM

np.linspace(0,o3pM,10)

np.linspace(0,o3pM,10)*2e3

[format(x,'.2f') if i<7 else "" for (i, x) in enumerate(cb3tlins*2.45e3)]

[format(x,'.4f') if i<7 else "" for (i, x) in enumerate(cb3tlins*2.5e3)]

np.max(only3phi(phi))

np.max(only3(psi))

pzlin[45]/hb/k

only3phi(phi)[45]

plt.plot(fucking_scaling_fuck(np.linspace(0,0.003,2000),1e-5,1e-6,1e4))



# +
# # t=0.03
# # for t in [0.03, 0.06, 0.09, 0.012, 0.15]:
# for t in [0.5657, 0.5857, 0.6057, 0.6257, 0.6347]:
# # for t in [t=1.2052]
#     print(f"exporting figures for t={round(t,5)}")
#     data_folder = "20240711-234819-TFF" #"20240528-224811-TFF"
#     del psi, phi, swnf
#     with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=8) as file:
#         psi = pickle.load(file)
#     phi, swnf = phiAndSWNF(psi, nthreads=7)
#     gc.collect()

#     psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True)

#     whatever_bla_testing1 = plot_g34(phi, cutPlot=1.5, saveFig=True, 
#             title2=f"Mirror Pulse",
#             title2filestr="MP", skipPlot=True
#             )

# +
# t=1.2052
# data_folder = "20240711-234819-TFF" #"20240528-224811-TFF"
# for settingStr in comboSettSetS:
#     del psi, phi, swnf
#     with pgzip.open(f'/Volumes/tonyNVME Gold/twoParticleSim/{data_folder}/psi at t={round(t,5)}.pgz.pkl', 'rb', thread=8) as file:
#         psi = pickle.load(file)
#     phi, swnf = phiAndSWNF(psi, nthreads=7)
#     gc.collect()

#     psi_phi_plot1(t,-1,psi,phi, plt_show=False, plt_save=True)

#     whatever_bla_testing1 = plot_g34(phi, cutPlot=1.5, saveFig=True, 
#             title2=f"Initial scattering",
#             title2filestr="NA", skipPlot=True
#             )
# -

pxmax/(hb*k)



psi, phi = None, None
gc.collect()



np.show_config()

np.test()

















# # Benchmarking

psi,phi = None,None
gc.collect()

import timeit

psi = psi0gaussian(sx3=5, sz3=5, sx4=5, sz4=5, px3=0, pz3=0, px4=0, pz4=0)
phi, swnf = phiAndSWNF(psi, nthreads=7)

bench_scattering_evolve_bragg_loop_helper2 = datetime.now()
_ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,steps=1,progress_proxy=None,s34=1e6,
                                        t3mid=-3,t3wid=1,v3pot=0,t4mid=0.5*t4sc,t4wid=t4sc,v4pot=V4sc)
l.info(f"scattering_evolve_bragg_loop_helper2 one run {(datetime.now()-bench_scattering_evolve_bragg_loop_helper2)}")

with ProgressBar(total=10) as progressbar: 
    bench_scattering_evolve_bragg_loop_helper2 = datetime.now()
    _ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,steps=10,progress_proxy=progressbar,s34=evolve_s34,
                                            t3mid=-3,t3wid=1,v3pot=0,t4mid=0.5*t4sc,t4wid=t4sc,v4pot=V4sc)
    l.info(f"scattering_evolve_bragg_loop_helper2 one run {(datetime.now()-bench_scattering_evolve_bragg_loop_helper2)}")

with ProgressBar(total=10) as progressbar: 
    _ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                                    steps=10,progress_proxy=progressbar,s34=0,
                                    t3mid=T_BS, t3wid=t3pi4, v3pot=V3pi4, 
                                    t4mid=T_BS, t4wid=t4pi4, v4pot=V4pi4,  
                                    numba_threads = 8,fftw_threads  = 8)

# + vscode={"languageId": "raw"} active=""
# M1 Max using accelerate
# init psi:       ??
# evolve a34=.    9.87
# evolve a34=0    10.57, 10.23
#
# M1 Max using noarch
# init psi        10.7, 9.3
# evolve a34=.    9.68, 8.41, 8.76
# evolve a34=0    8.49, 7.43, 7.37
#
#
# RSPE 11700
# init psi:       41.3,  24.6
# evolve a34=.    19.92, 18.63
# evolve a34=0    16.44, 16.98
#
#
# -
x

bench_scan_bencht_scattering_evolve_bragg_loop_helper2 = np.zeros((16,16))
for numba_threads in range(1,17):
    for fftw_threads in  range(1,17):
        print(f"numba_threads={numba_threads}, fftw_threads={fftw_threads}")
        bencht_scattering_evolve_bragg_loop_helper2 = datetime.now()
        with ProgressBar(total=10) as progressbar: 
            _ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                                            steps=10,progress_proxy=progressbar,s34=0,
                                            t3mid=T_BS, t3wid=t3pi4, v3pot=V3pi4, 
                                            t4mid=T_BS, t4wid=t4pi4, v4pot=V4pi4,  
                                            numba_threads = numba_threads,fftw_threads  = fftw_threads)
        bench_t_scan_delta = datetime.now()-bencht_scattering_evolve_bragg_loop_helper2
        print(f"    bench_t_scan_delta = {bench_t_scan_delta}")
        # bench_scan_bencht_scattering_evolve_bragg_loop_helper2.append((numba_threads,fftw_threads,bench_t_scan_delta))
        bench_scan_bencht_scattering_evolve_bragg_loop_helper2[numba_threads-1,fftw_threads-1] = bench_t_scan_delta.total_seconds()

bench_scan_bencht_scattering_evolve_bragg_loop_helper2 = []
for numba_threads in [1,2,4,6,8,10,12,14]:
    for fftw_threads in  [1,2,4,6,8,10,12,14]:
        print(f"numba_threads={numba_threads}, fftw_threads={fftw_threads}")
        bencht_scattering_evolve_bragg_loop_helper2 = datetime.now()
        with ProgressBar(total=10) as progressbar: 
            _ = scattering_evolve_bragg_loop_helper2(t,psi,swnf,
                                            steps=10,progress_proxy=progressbar,s34=0,
                                            t3mid=T_BS, t3wid=t3pi4, v3pot=V3pi4, 
                                            t4mid=T_BS, t4wid=t4pi4, v4pot=V4pi4,  
                                            numba_threads = numba_threads,fftw_threads  = fftw_threads)
        bench_t_scan_delta = datetime.now()-bencht_scattering_evolve_bragg_loop_helper2
        print(f"    bench_t_scan_delta = {datetime.now()-bencht_scattering_evolve_bragg_loop_helper2}")
        bench_scan_bencht_scattering_evolve_bragg_loop_helper2.append((numba_threads,fftw_threads,bench_t_scan_delta))


bench_scan_bencht_scattering_evolve_bragg_loop_helper2

df_bench = pd.DataFrame(bench_scan_bencht_scattering_evolve_bragg_loop_helper2, columns=['numba_threads', 'fftw_threads', 'time_delta'])
df_bench['time_seconds'] = df_bench['time_delta'].apply(lambda x: x.total_seconds())
df_bench.drop(columns=['time_delta'], inplace=True)
print(df_bench)

df_bench['time_seconds'].min()

min_time_row = df_bench.loc[df_bench['time_seconds'].idxmin()]
print(min_time_row)







# + vscode={"languageId": "raw"} active=""
# @jit(forceobj=True, cache=True)
# def phiAndSWNF(psi, nthreads=nthreads):
#     phiUN = np.flip(np.flip(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(psi,threads=nthreads,norm='ortho')),axis=1),axis=3)
#     superWeirdNormalisationFactorSq = check_norm_phi(phiUN)
#     swnf = sqrt(superWeirdNormalisationFactorSq)
#     phi = phiUN/swnf
#     return phi, (swnf+0*1j)
#
# @jit(forceobj=True, cache=True)
# def toPhi(psi, swnf, nthreads=nthreads) -> np.ndarray:
#     return np.flip(np.flip(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(psi,threads=nthreads,norm='ortho')),axis=1),axis=3)/swnf
#
# @jit(forceobj=True, cache=True)
# def toPsi(phi, swnf, nthreads=nthreads) -> np.ndarray:
#     return pyfftw.interfaces.numpy_fft.ifftn(np.fft.ifftshift(np.flip(np.flip(phi*swnf,axis=3),axis=1)),threads=nthreads,norm='ortho')
# -

import cupy as cp


# +
# Conversion functions
def numpy_to_cuda(array: np.ndarray) -> cp.ndarray:
    return cp.asarray(array)

def cuda_to_numpy(array: cp.ndarray) -> np.ndarray:
    return cp.asnumpy(array)

# CUDA-compatible normalization check
def check_norm_phi_cuda(phi_cuda):
    # Assuming uniform spacing, replace trapz with sum approximation
    norm = cp.sum(cp.abs(phi_cuda)**2) * (dpx * dpx * dpz * dpz)
    return norm

# CUDA-compatible functions
def phiAndSWNF_cuda(psi):
    phiUN = cp.flip(cp.flip(cp.fft.fftshift(cp.fft.fftn(psi, norm='ortho'), axes=(1, 3)), axis=1), axis=3)
    superWeirdNormalisationFactorSq = check_norm_phi_cuda(phiUN)
    swnf = cp.sqrt(superWeirdNormalisationFactorSq)
    phi = phiUN / swnf
    return phi, swnf

def toPhi_cuda(psi, swnf) -> cp.ndarray:
    phi = cp.flip(cp.flip(cp.fft.fftshift(cp.fft.fftn(psi, norm='ortho'), axes=(1, 3)), axis=1), axis=3) / swnf
    return phi

def toPsi_cuda(phi, swnf) -> cp.ndarray:
    psi = cp.fft.ifftn(cp.fft.ifftshift(cp.flip(cp.flip(phi * swnf, axis=3), axis=1)), norm='ortho')
    return psi



# -

psi = psi0gaussian(sx3=su, sz3=su, sx4=su, sz4=su, px3=0, pz3=0, px4=0, pz4=0)

phi, swnf = phiAndSWNF(psi, nthreads=7)

psi_cuda = numpy_to_cuda(psi)

phi_cuda, swnf_cuda = phiAndSWNF_cuda(psi_cuda)

phi_cuda.shape



# +
# 1. Delete GPU variables
psi_cuda = None
phi_cuda = None

# 2. Free CuPy's default memory pool
cp.get_default_memory_pool().free_all_blocks()

# (Optional) Free CuPy's pinned memory pool if used
cp.get_default_pinned_memory_pool().free_all_blocks()

# 3. Run garbage collection
gc.collect()
# -

cp.cuda.Device(0).reset()

cp.cuda.Stream.null.synchronize()

cp.cuda.set_allocator(None)
