# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:51:29 2023

@author: sfs135
"""

# %% Load libraries

from pyFDTD import pyFDTD
import matplotlib.pyplot as plt
import numpy as np
import time

# %% Get number of dimensions as input

import sys
if len(sys.argv) > 1:
    NDim = int(sys.argv[1])
else:
    NDim = 2

# %% Run pyFDTD

start_time = time.time()

fdtd = pyFDTD(NDim=NDim, debug=True)

fdtd.t = 10e-3
fdtd.X = 10e-3
fdtd.setInputs()

fdtd.plotIthUpdate = 5
fdtd.doPlot = True
fdtd.saveRecData = False
fdtd.saveGif = False

if fdtd.NDim == 1:
    # 1D Mesh
    fdtd.betaMode = 'varying'
    fdtd.mesh[0:10] = 1
    fdtd.mesh[90:] = 0.5
elif fdtd.NDim == 2:
    # # 2D mesh
    # fdtd.newEmptyMesh([200, 300])
    # fdtd.moveSrc([1.0, 0.75])
    # fdtd.srcFreq[0] = 1200
    # Load a 2D image
    fdtd.betaMode = 'varying'
    fdtd.betaBorder = 1
    # fdtd.loadImage('../images/Flat_wall.png')
    fdtd.loadImage('../images/Binary_amplitude_diffuser.png')
    fdtd.moveSrc([1.25, 1.25])
    fdtd.moveRec([1.25, 1.75])
elif fdtd.NDim == 3:
    # 3D mesh
    fdtd.newEmptyMesh([120, 170, 50])
    # Place a block in the mesh
    fdtd.mesh[60:100,70:100,20:40] = 1
    # Move the source
    fdtd.moveSrc([0.5, 0.6, 0.3])

# Run simulation
fdtd.run()

print("--- %s seconds ---" % (time.time() - start_time))

# %%

input('FINISHED!!\nPress key to exit.')

# %%

plt.close(fdtd.hf)

# %%

