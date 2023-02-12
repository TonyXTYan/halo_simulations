import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ["MKL_NUM_THREADS"] = '6' # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = '6' # export NUMEXPR_NUM_THREADS=6


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



warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
# warnings.filterwarnings("ignore", category=UserWarning) 
warnings.formatwarning = lambda s, *args: "Warning: " + str(s)+'\n'

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.close("all") # close all existing matplotlib plots
# plt.ion()        # interact with plots without pausing program



from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



from numba import njit, jit, prange, objmode, vectorize
import numba
numba.set_num_threads(8)

from numba_progress import ProgressBar

from matplotlib.ticker import MaxNLocator

import gc
gc.enable(  )
# gc.set_debug(gc.DEBUG_SAVEALL)




dtype = np.cdouble
dtyper = np.float64


use_cache = False
datatime_now = datetime.now()
output_prefix_bracket = "("+datatime_now.strftime("%Y%m%d-%H%M%S") + "-" + \
                        str("T" if use_cache else "F") + ")"
output_prefix = "output/entangled_halo_proj/"+output_prefix_bracket+" "
output_ext = ".pgz.pkl"
print(output_prefix)


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
        
        
        
nx = 250+1
nz = 80+1
xmax = 25 # Micrometers
# xmax = 20 * 0.001 # Millimeters
# zmax = (nz/nx)*xmax
zmax = 22
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


pxmax= (nx+1)/2 * 2*pi/(2*xmax)*hb # want this to be greater than p
pzmax= (nz+1)/2 * 2*pi/(2*zmax)*hb

dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb

wavelength = 1.083 #Micrometers
# k = (1/sqrt(2)) * 2*pi / wavelength # effective wavelength
# k = 0.03 * 2*pi / wavelength
k = pi / (4*dx)
# k = pi / (2*dz)
p = 2*hb*k

v3 = hb*k/m3
v4 = hb*k/m4