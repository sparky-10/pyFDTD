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

# %% Run pyFDTD

start_time = time.time()

fdtd = pyFDTD(NDim=2, debug=True)
fdtd.t = 5e-3
fdtd.newEmptyMesh([400,400])

# Some run settings
fdtd.plotIthUpdate = 2
fdtd.doPlot = True
fdtd.saveRecData = True
fdtd.saveGif = False

# Before run and/or prepareMesh
fdtd.beta = 0.0
fdtd.betaBorder = 1.0
fdtd.threshold = 0.0
fdtd.betaMode = 'constant'
# fdtd.betaMode = 'varying'

# fdtd.loadImage('../images/Flat_wall.png')
# # fdtd.image2Mesh(xMesh=200, yMesh=200);
# fdtd.moveSrc([fdtd.D[0]*0.5, fdtd.D[1]*0.7],0)
# fdtd.moveRec([fdtd.D[0]*0.5, fdtd.D[1]*0.85],0)

# # fdtd.newEmptyMesh([50,50,50])
# # fdtd.setInputs()
# fdtd.mesh[20:30,20:30,10:40] = 1
# fdtd.cLims = [-0.02, 0.02]
# fdtd.moveSrc([fdtd.D[0]*0.3, fdtd.D[1]*0.4, fdtd.D[2]*0.1],0)


# fdtd.addSrc([fdtd.X*10, fdtd.X*10, fdtd.X*10])


# Run simulation in steps
# fdtd.runReset()
# fdtd.checkMesh()
# fdtd.prepareMesh()
#for i in range(0,50):
#    fdtd.runSteps()
#    print(fdtd.running)

# Run full simulation
fdtd.run()

print("--- %s seconds ---" % (time.time() - start_time))

# %%

input('FINISHED!!\nPress key to exit.')

# %%

plt.close(fdtd.hf)

# %%

