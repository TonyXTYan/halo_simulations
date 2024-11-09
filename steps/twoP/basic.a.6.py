
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
os.environ["IMAGEIO_FFMPEG_EXE"]="/envs/py311_he34sim/bin/ffmpeg"
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
        plt.savefig(output_prefix+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_prefix+title+".png", dpi=600, bbox_inches='tight')
    if skipPlot:
        plt.close()
    else:
        plt.show()
    return (gx3px4p, gx3px4m, gx3mx4p, gx3mx4m, gx3x4n)


def plot_g34_v2(phiHere, cutPlot=1.5, saveFig=True, 
             pMaxCut=2,
             title2="", title2filestr="NA",
             skipPlot=False,
             figSize=(9,3.5)):
    gx3px4p = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=+p,offset4=+p)
    gx3px4m = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=+p,offset4=-p)
    gx3mx4p = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=-p,offset4=+p)
    gx3mx4m = gp3p4_dhalo_calc(phiHere,cut=cutPlot,offset3=-p,offset4=-p)
    
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
    plt.figure(figsize=figSize)
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
    plt.colorbar(im,pad=0.15,fraction=0.046)

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
    cbar = plt.colorbar(im,pad=0.15,fraction=0.046)
    # cbar.set_ticks(np.linspace(0,0.25,6))

    title = f"CorrE t={round(t,5)}, cut = {cutPlot}, s={title2filestr}"
    if saveFig:
        plt.savefig(output_prefix+title+".pdf", dpi=600, bbox_inches='tight')
        plt.savefig(output_prefix+title+".png", dpi=600, bbox_inches='tight')
    if skipPlot:
        plt.close()
    else:
        plt.show()
    return (gx3px4p, gx3px4m, gx3mx4p, gx3mx4m, gx3x4n)

