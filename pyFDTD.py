
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('wxAgg')
# matplotlib.use('Qt5Agg')

import numpy as np
from PIL import Image
# import time
            
# Constants
PI = np.pi

# Start settings
NDIM = 2
NXYZ = [100, 150, 50]
X_STEP = 0.01
SRC_XYZ = [[0.25, 0.75, 0.1]]
REC_XYZ = [[0.75, 0.25, 0.2]]
SRC_TYPE_DEFAULT = "Gauss deriv 1"
C0 = 344
SIM_TIME = 0.01
PLOT_UPDATE_TIME = 0.01
PLOT_ITH_UPDATE = 1
DO_PLOT = True
DEBUG = False
DEBUG_PREFIX = ""

USE_SCIPY = True        # WARNING - set to False is limited and untested
USE_MATPLOTLIB = True
        
# If using scipy or not (for 2D convolve)
if USE_SCIPY:
    from scipy.signal import convolve2d, resample
    from scipy import ndimage
    from scipy.io import wavfile
else:
    # TODO - non-scipy wav write
    ()

if USE_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
            
# TODO

# GIF
# - Test 3D vertical orientation
# - Do 1D
# - Add src/recs??
# Test 1D and 3D
# - Sim
# - 1D no negative component??
# - Mesh creating
# Plots:
# - Add src/rec labels
# - Add optional plotSliceHeight for 3D
# Do staggered mesh to grid
# Add proper examples
# Do proper readme
# Add more debug text?
# Improve mesh fix (diagonals causing dispersion??)
# Alter src/rec to a dict - e.g. srcXyz -> src['Xyz']

# Future:
# Filter surfaces
# Thread??

class pyFDTD:
    
    def __init__(self, 
                 NDim=NDIM,
                 debug=DEBUG, 
                 debugPrefix=DEBUG_PREFIX, 
                 doPlot=DO_PLOT):
        
        # Init
        
        # No. dimensions, debug and plot modes
        self.NDim = NDim
        self.debugPrefix = debugPrefix
        self.debug = debug
        self.doPlot = doPlot
        
        # Update on debug
        self.printToDebug('__init__')
        
        # Define defaults
        self.setDefaults()
        
        # Set basic inputs
        self.setInputs()
        
        # Default empty mesh
        self.meshReset(doPrepareMesh=True)
        
        # Default source and receivers
        for src in SRC_XYZ:
            self.addSrc(src[0:self.NDim])
        for rec in REC_XYZ:
            self.addRec(rec[0:self.NDim])
    
    def setDefaults(self):
        
        # Update on debug
        self.printToDebug('setDefaults')
        
        # Defaults
        self.c                  = C0
        self.X                  = X_STEP
        self.t                  = SIM_TIME
        self.plotUpdateTime     = PLOT_UPDATE_TIME
        self.plotIthUpdate      = PLOT_ITH_UPDATE
        
        self.saveRecData        = False
        self.recDataFile        = 'recData.wav'
        self.fsOut              = None
        
        self.saveGif            = False
        self.gifFile            = 'gifOut.gif'
        self.gifFrameTime       = 40    # ms
        self.gifLoopNum         = 0
        self.gifArea            = None
        
        self.beta               = 0
        self.betaBorder         = 0
        self.betaType           = 'admittance'
        self.betaMode           = 'constant'
        self.cLims              = [-0.1, 0.1]
        self.c0                 = [1.0, 1.0, 1.0]
        self.c1                 = [0.0, 0.0, 1.0]
        self.c2                 = [1.0, 0.0, 0.0]
        self.da                 = 0
        self.db                 = 0
        self.dc                 = 0
        
        self.imageThreshold     = 0
        self.plotShowMask       = True
        self.plotMaskColInvert  = False
        self.plotShowColBar     = True
        self.plotShowSrc        = True
        self.plotShowRec        = True
        self.image              = None
        self.mesh               = None
        
        self.doSrcRecCoordsCheck \
                                = True
        
        self.srcXyz             = []
        self.recXyz             = []
        self.srcXyzDisc         = []
        self.recXyzDisc         = []
        self.srcData            = []
        self.recData            = []
        self.srcInd             = []
        self.recInd             = []
        self.srcFlatInd         = []
        self.recFlatInd         = []
        self.srcNodeType        = []
        self.recNodeType        = []
        self.srcAmp             = []
        self.recAmp             = []
        self.srcStrength        = []
        self.recStrength        = []
        self.srcT0              = []
        self.srcFreq            = []
        self.srcType            = []
        self.srcN               = len(self.srcXyz)
        self.recN               = len(self.recXyz)
        
        self.psn,   self.pon,   self.pgn    = None, None, None
        self.psn1,  self.psn2,  self.psn3   = None, None, None
        self.pon1,  self.pon2,  self.pon3   = None, None, None
        self.bp1_1, self.bp1_2, self.bp1_3  = None, None, None
        self.bm1 = None
        
        self.updatePlotThisLoop = None
        self.running            = False
        self.figNum             = None
        
        # Default grid size
        self.Nxyz = NXYZ[0:self.NDim]
        
        # Adjust colour plot limits according to No. dimensions
        cLimScale = (1/40)**(self.NDim-2)
        self.cLims = [cLim*cLimScale for cLim in self.cLims]
        
    def run(self):
        
        # Run full simulation
        
        # Update on debug
        self.printToDebug('run')
        
        # Ensure basic inputs are up to date
        self.setInputs()
        
        # Sets back to start of sim
        self.runReset()
        
        # Check mesh
        self.checkMesh()
        
        # Prepare mesh
        self.prepareMesh()
        
        # Loop round until end of sim
        self.runSteps(self.Nt)
        
    def runSteps(self, nSteps = None):
        
        # Run specified number of steps
        # No setting of inputs, resetting sim etc.
        
        # Default to plotIthUpdate number of steps
        if nSteps == None:
            nSteps = self.plotIthUpdate
        
        # Do to end loop number
        endLoopNum = self.loopNum+nSteps
        endLoopNum = min(endLoopNum, self.Nt)
        
        # # Check loop number (determines state of self.running)
        # self.checkLoopNum()
        
        # Loop...
        while self.loopNum < endLoopNum:
            
            # Break if not running
            # Note: currently does nothing - need to do in different thread 
            # for this to have an effect
            if self.loopNum > 0 and not self.running:
                return self.running
            
            # Run simulation step
            self.runStep()
            
        return self.running
            
    def runStep(self):
        
        # Run single simulation step
        # No setting of inputs, resetting sim etc.
        
        # Single run step
        i = self.loopNum
        
        # Where we're up to (1-indexed)
        self.printToDebug("runStep %i/%i"%(self.loopNum+1,self.Nt))
        
        # Update plot on this loop
        self.updatePlotThisLoop = (self.loopNum+1)%self.plotIthUpdate == 0
        
        if i >= self.Nt:
            
            # Finished
            self.running = False
            return self.running
        
        elif i <= 0:
            
            # Set running to true if on firt loop
            self.running = True
            
        else:
            
            # Do grid update on i > 0
            
            # Multiply previous surface pressures by beta-1
            
            self.pz.flat[self.psn] *= self.bm1
            
            # Swap p and pz
            self.p, self.pz = self.pz, self.p
            
            # Convolve with 'a' matrix, while swapping p and pz back again
            self.p *= -1
            if USE_SCIPY:
                if self.NDim == 1:
                    self.p += np.convolve(self.pz,self.a,mode='same')
                elif self.NDim == 2:
                    # Because faster than ndimage.convolve
                    self.p += convolve2d(self.pz, self.a, mode='same')
                elif self.NDim == 3:
                    self.p += ndimage.convolve(self.pz, self.a, mode='constant')
            else:
                # Numpy equivalent assuming a is expected 2D matrix
                # (i.e. [[0,0.5,0], [0.5,0,0.5], [0,0.5,0]])
                for k in range(0,self.Nxyz[0]):
                    self.p[k,:] += np.convolve(self.pz[k,:],self.a[1,:],mode='same')
                for k in range(0,self.Nxyz[1]):
                    self.p[:,k] += np.convolve(self.pz[:,k],self.a[1,:],mode='same')
            
            # Sum of prev pressures at opp nodes
            sumPz1 = self.pz.flat[self.pon1]
            sumPz2 = self.pz.flat[self.pon2[:,0]] + self.pz.flat[self.pon2[:,1]]
            sumPz3 = self.pz.flat[self.pon3[:,0]] + self.pz.flat[self.pon3[:,1]] + \
                self.pz.flat[self.pon3[:,2]]
            
            # Add pressure from 'opposite' nodes
            self.p.flat[self.psn1] = (self.p.flat[self.psn1] + self.d1*sumPz1) * self.bp1_1           # Surfaces
            self.p.flat[self.psn2] = (self.p.flat[self.psn2] + self.d1*sumPz2) * self.bp1_2    # Edges
            self.p.flat[self.psn3] = (self.p.flat[self.psn3] + self.d1*sumPz3) * self.bp1_3    # Corners
            
            # Set 'ghosts' back to zero
            self.p.flat[self.pgn] = 0.0
        
        # Add source pressure
        for j in range(0,self.srcN):
            self.p.flat[self.srcFlatInd[j]] += \
                self.srcStrength[j] * self.srcData[j][i]
        
        # Store receiver pressure
        for j in range(0,self.recN):
            self.recData[j][i] = \
                self.recStrength[j] * self.p.flat[self.recFlatInd[j]]
        
        # Update plot
        self.updatePlot()
        
        # Update loop number or finish
        if i >= self.Nt-1:
            # This was the last loop
            self.running = False
            # Save result
            if self.saveRecData:
                self.writeRecData()
            # Save GIF
            if self.saveGif:
                self.writeGifData()
                    
        elif self.running:
            # Increment loop number if still running
            self.loopNum +=1
        
        return self.running
            
    def runReset(self):
        
        # Reset sim to beginning
        
        # Update on debug
        self.printToDebug('runReset')
        
        # Set pressure field to zeros
        self.pReset()
        
        # Reset all src/receiver data
        self.srcRecDataReset()
        
        # Clear plot
        #self.updatePlot(True)
        self.makePlot()
        
        # Reset loop count
        self.loopNum = 0        
    
    def stop(self):
        
        # Stop sim
        
        # Update on debug
        self.printToDebug('stop')
            
        # Stop simulation
        self.running = False
        
    def checkLoopNum(self):
        
        # Check loop number - don't think used anymore
        
        # Check loop num to see if valid
        if self.loopNum == 0:
            # Start of sim
            self.running = True
        elif self.loopNum == self.Nt:
            # End of sim
            self.running = False
    
    def setInputs(self):
        
        # Set the basic inputs
        
        # Update on debug
        self.printToDebug('setInputs')
        
        # Stop if basic inputs are redefined
        self.stop()
        
        # Courant No. and update coeffs
        self.courantNo()
        self.updateCoeffs()
        self.getConvMat()
        
        # Sample rate, time step, and new speed of sound
        self.fs = round(self.c/(self.lam*self.X))
        self.T = 1/self.fs
        self.c = self.X*self.lam/self.T
        
        # Size of space
        self.NDim = len(self.Nxyz)
        self.D = [(N-1)*self.X for N in self.Nxyz]
        # Total problem size
        self.Ntot = np.prod(self.Nxyz)
        
        # Number time steps
        self.Nt = int(np.ceil(self.t/self.T))
        
        # check coordinates of source/receivers
        if self.doSrcRecCoordsCheck:
            self.checkSrcRecCoords()
        
        # Set colour map
        self.setCMap()
    
    def courantNo(self):
        
        # Define Courant No.
        self.lam2 = [1-4*self.da, (1-8*self.da+16*self.da**2)/(2-4*self.db), \
            (1-12*self.da+48*self.da**2-64*self.da**3)/ \
                (3-12*self.db+16*self.dc)]
        self.lam2 = np.min(self.lam2[0:self.NDim])
        
        self.lam = np.sqrt(self.lam2)
    
    def updateCoeffs(self):
        
        # Calculate update coefficients
        dims = [0]*3
        for i in range(0,self.NDim):
            dims[i] = 1
        lambda2 = self.lam2
        self.d1 = lambda2*(1-2*(self.NDim-1)*self.db+dims[2]*4*self.dc)
        self.d2 = lambda2*(dims[1]*self.db-dims[2]*2*self.dc)
        self.d3 = lambda2*dims[2]*self.dc
        self.d4 = 2*(1-self.NDim*lambda2+ \
                     dims[1]*((self.NDim-1)**2+self.NDim-1)*self.db*lambda2- \
                     dims[2]*4*self.dc*lambda2)
    
    def getConvMat(self):
        
        # Make convolutin matrix
        if self.NDim == 1:
            self.a = np.zeros((3))
            self.a[[0,2]] = self.d1
            self.a[1] = self.d4
        elif self.NDim == 2:
            self.a = np.zeros((3,3))
            self.a[1,[0,2]] = self.d1
            self.a[[0,2],1] = self.d1
            self.a[[0,2],0] = self.d2
            self.a[[0,2],2] = self.d2
            self.a[1,1] = self.d4
        elif self.NDim == 3:
            self.a = np.zeros((3,3,3))
            self.a.flat[np.array([4,10,12,14,16,22])] = self.d1
            self.a.flat[np.arange(1,12,2)] = self.d2
            self.a.flat[np.arange(15,26,2)] = self.d2
            self.a.flat[np.array([0,2,6,8,18,20,24,26])] = self.d3
            self.a[1,1,1] = self.d4
    
    def checkSrcRecCoords(self):
        
        # Check source and receiver coordinates
        # Delete if outside area and move if not
        for i in range(0,self.srcN):
            ii = self.srcN-i-1  # Go backwards
            if not self.checkCoords(self.srcXyz[ii]):
                self.delSrc(ii)
            else:
                self.moveSrc(self.srcXyz[ii], ii)
        for i in range(0,self.recN):
            ii = self.recN-i-1  # Go backwards
            if not self.checkCoords(self.recXyz[ii]):
                self.delRec(ii)
            else:
                self.moveRec(self.recXyz[ii], ii)
    
    def checkCoords(self, xyz):
        
        # Check if coordinates are in range
        if len(xyz) != self.NDim:
            return False
        else:
            coordsCheck = True
            for i, x in enumerate(xyz):
                coordsCheck = coordsCheck and x >= 0 and x <= self.D[i]
        
        return coordsCheck
    
    def pReset(self):
        
        # Set pressure field to zeros
        self.p = np.zeros((self.Nxyz))
        self.pz = np.zeros((self.Nxyz))
    
    def checkMesh(self, fix=True):
        
        # Check mesh (and attempt to fix if specified)
        
        # Update on debug
        self.printToDebug('checkMesh')
        
        # Grid size (this should already have happened)
        self.Nxyz = list(self.mesh.shape)
        
        # No issues found unless find otherwise
        anyIssues = False
        
        # Counters
        dim = 0;
        consecDims = 0;
        
        # Loop round dimensions until no more changes
        while consecDims < self.NDim:
            
            # Size of current dimension
            Ni = self.Nxyz[dim]
            
            # Indices to either side of nodes
            inds1 = np.abs(np.arange(Ni)-1)
            inds2 = np.flip(Ni-1-inds1)
            indShape = np.ones(self.NDim, dtype=int)
            indShape[dim] = Ni
            inds1 = np.reshape(inds1, indShape)
            inds2 = np.reshape(inds2, indShape)
            # get neighbours
            neighbours1 = np.take_along_axis(self.mesh, inds1, axis=dim)
            neighbours2 = np.take_along_axis(self.mesh, inds2, axis=dim)
            # Find surrounding surfaces around single air nodes
            invalid = np.logical_or(neighbours1==0, neighbours2==0)
            invalid = np.logical_or(invalid, self.mesh!=0)
            neighbours1[invalid] = 0
            neighbours2[invalid] = 0
            # Combine and average (if floats, otherwise not needed)
            if self.mesh.dtype.kind == 'f':
                neighbours1 += neighbours2
                neighbours1 *= 0.5
            
            # Any nodes found that need fixing?
            foundNodes = np.any(neighbours1>0)
            
            # No change occurred unless find otherwise
            anyChange = False
            
            if foundNodes:
                # Houston, we have a problem
                anyIssues = True
                if fix:
                    self.mesh += neighbours1
                    # Changes have been made
                    anyChange = True
            
            # Update consecutive dimensions with no change
            if anyChange:
                # Reset
                consecDims = 0
            else:
                # Increase
                consecDims += 1
            
            # Next dimension to look at
            dim = np.remainder(dim+1, self.NDim)
        
        return anyIssues
    
    def prepareMesh(self, betaMode = None):
        
        # Update on debug
        self.printToDebug('prepareMesh')
            
        # Make update parameters for grid from numpy array
        # WARNING: by no means fool proof and/or well tested!!!
        
        # Admittance type
        if betaMode is None:
            betaMode = self.betaMode
        
        # Grid size (this should already have happened)
        self.Nxyz = list(self.mesh.shape)
        
        # Copy of input as zeros and ones
        nodes = (self.mesh>0).astype(int)
        
        # Dimensions
        dims = np.arange(self.NDim).astype(int)
        Ni = np.array(self.Nxyz).astype(int)
        
        # Empty lists
        self.pgn = np.empty((0)).astype(int)
        self.psn = np.empty((0)).astype(int)
        self.pon = np.empty((0)).astype(int)
        beta = np.empty((0))
        
        # Loop round dimensions
        for i in range(0,self.NDim):
            
            # Differences in positive and negative direction along i-th dimension
            diff1 = np.diff(nodes,n=1,append=1)
            diff2 = np.flip(np.diff(np.flip(nodes, axis=-1),n=1,append=1), axis=-1)
            
            # Flatten
            diff1 = diff1.flatten()
            diff2 = diff2.flatten()
            
            # Surface node indices
            self.psn1 = np.where(diff1==1)[0]
            self.psn2 = np.where(diff2==1)[0]
            
            # 'Opposite' node indices
            self.pon1 = self.psn1-1
            self.pon2 = self.psn2+1
            
            # # 'Ghost' node indices
            # pgn1 = np.where(diff1<0)[0]
            # pgn2 = np.where(diff2<0)[0]
            
            # 'Ghost' node indices
            pgn1 = self.psn1+1
            pgn2 = self.psn2-1
            
            # Ghosts
            # If surfaces were first or last and hence ghosts would be in a 
            # different dimension(/column etc.)
            isFirstIndex = np.remainder(pgn1,Ni[-1])==0
            isLastIndex = np.remainder(self.psn2,Ni[-1])==0
            
            # Default to beta on border
            if hasattr(self.betaBorder, "__len__"):
                i1 = (dims[-1]*2)%len(self.betaBorder)
                i2 = (dims[-1]*2+1)%len(self.betaBorder)
                # Note switch in i1 and i2 as my 1s and 2s were defined the
                # other way round when I first wrote this
                betaBorder1 = self.betaBorder[i2]
                betaBorder2 = self.betaBorder[i1]
            else:
                betaBorder1 = self.betaBorder
                betaBorder2 = self.betaBorder
            beta1 = np.ones(len(pgn1))*betaBorder1
            beta2 = np.ones(len(pgn2))*betaBorder2
            
            # And then if not on border...
            if betaMode == 'constant':
                beta1[isFirstIndex==False] = self.beta
                beta2[isLastIndex==False] = self.beta
            elif betaMode == 'varying':
                # Admittance on ghost nodes
                inds = self.indFixDims(pgn1[isFirstIndex==False], Ni, dims)
                beta1[isFirstIndex==False] = 1-self.mesh.flat[inds]
                inds = self.indFixDims(pgn2[isLastIndex==False], Ni, dims)
                beta2[isLastIndex==False] = 1-self.mesh.flat[inds]
            # Get rid of ghosts at border
            pgn1 = pgn1[isFirstIndex==False]
            pgn2 = pgn2[isLastIndex==False]
            
            # Opposites
            # If surfaces were first or last and hence opposites would be in a 
            # different dimension(/column etc.)
            isFirstIndex = np.remainder(self.psn1,Ni[-1])==0
            isLastIndex = np.remainder(self.pon2,Ni[-1])==0
            
            # Check nodes are in same dimension and an 'air' node
            ind_check = np.logical_or(isFirstIndex, nodes.flat[self.pon1]!=0)
            if ind_check.any():
                self.pon1 = self.pon1[ind_check==False]
                self.psn1 = self.psn1[ind_check==False]
                #pgn1 = pgn1[ind_check==False] # It's still a ghost node!
                beta1 = beta1[ind_check==False]
            ind_check = np.logical_or(isLastIndex, nodes.flat[self.pon2]!=0)
            if ind_check.any():
                #pgn2 = psn2[ind_check]
                self.pon2 = self.pon2[ind_check==False]
                self.psn2 = self.psn2[ind_check==False]
                #pgn2 = pgn2[ind_check==False] # It's still a ghost node!
                beta2 = beta2[ind_check==False]
                
            # Concatenate lists
            self.psn1 = np.append(self.psn1, self.psn2)
            self.pon1 = np.append(self.pon1, self.pon2)
            pgn1 = np.append(pgn1, pgn2)
            beta1 = np.append(beta1, beta2)
            
            # Convert to correct indices (to account for shifting dimensions at
            # beginning)
            self.psn1 = self.indFixDims(self.psn1, Ni, dims)
            self.pon1 = self.indFixDims(self.pon1, Ni, dims)
            pgn1 = self.indFixDims(pgn1, Ni, dims)
            
            # Add to lists
            self.psn = np.append(self.psn, self.psn1)
            self.pon = np.append(self.pon, self.pon1)
            self.pgn = np.append(self.pgn, pgn1)
            beta = np.append(beta, beta1)
            
            # shift dimensions
            oldDims = dims
            dims = np.roll(dims,1)
            Ni = np.roll(Ni,1)
            nodes = np.moveaxis(nodes,dims,oldDims)
        
        # Sort order
        indSort = np.argsort(self.psn)
        self.psn = self.psn[indSort]
        self.pon = self.pon[indSort]
        beta = beta[indSort]
        
        # Convert to admittance if beta values are NIAC
        if self.betaType == 'absorption':
            beta = self.abs2Admit(beta)
        
        # Only keep unique ghost nodes
        self.pgn = np.unique(self.pgn)
        
        # Find surfaces occuring 1-3 times
        psn_unique, psn_uni_n, psn_uni_count = np.unique(self.psn, 
                                             return_index=True, 
                                             return_counts=True)
        self.psn1 = psn_unique[psn_uni_count==1]
        self.psn2 = psn_unique[psn_uni_count==2]
        self.psn3 = psn_unique[psn_uni_count==3]
        
        # Find opposites and admittances - plain surfaces
        #Ns1 = psn1.size
        n = psn_uni_n[psn_uni_count==1]
        self.pon1 = self.pon[n]
        b1 = beta[n]
        
        # Find opposites and admittances - edges
        Ns2 = self.psn2.size
        self.pon2 = np.zeros((Ns2,2), dtype=int)
        b2 =  np.zeros((Ns2,2))
        for i in range(Ns2):
            n = np.where(self.psn==self.psn2[i])
            self.pon2[i,:] = self.pon[n]
            b2[i,:] = beta[n]
        
        # Find opposites and admittances - corners
        Ns3 = self.psn3.size
        self.pon3 = np.zeros((Ns3,3), dtype=int)
        b3 =  np.zeros((Ns3,3))
        for i in range(Ns3):
            n = np.where(self.psn==self.psn3[i])
            self.pon3[i,:] = self.pon[n]
            b3[i,:] = beta[n]
        
        # Concatenated surface pressures
        self.psn = np.concatenate((self.psn1,self.psn2,self.psn3))
        
        # Beta+1 and beta-1
        self.bp1_1 = self.lam*b1+1
        self.bp1_2 = self.lam*b2.sum(axis=-1)+1
        self.bp1_3 = self.lam*b3.sum(axis=-1)+1
        self.bm1 = np.concatenate((self.bp1_1,self.bp1_2,self.bp1_3))-2
        
        # Use inverse of beta+1s (avoids division)
        self.bp1_1 = 1/self.bp1_1
        self.bp1_2 = 1/self.bp1_2
        self.bp1_3 = 1/self.bp1_3
        
        # Another speed up alteration
        self.bm1 *= -1
        
        # Source on surface amplitude factor
        self.srcSurfAmp = [1]*self.srcN
        
        for i in range(self.srcN):
            # Flat index to source
            self.srcFlatInd[i] = np.ravel_multi_index(self.srcInd[i], self.Nxyz)
            # Find if on non-air node
            self.srcNodeType[i] = nodes.flat[self.srcFlatInd[i]]
            if self.srcNodeType[i]:
                # If src ind is on mesh (non-air) node then set to zero
                self.srcStrength[i] = 0
            else:
                # Otherwise find if on surface type
                n1 = self.srcFlatInd[i]==self.psn1
                n2 = self.srcFlatInd[i]==self.psn2
                n3 = self.srcFlatInd[i]==self.psn3
                if np.any(n1):
                    n1 = np.where(n1)[0][0]
                    self.srcSurfAmp[i] = 2/(1+b1[n1])
                elif np.any(n2):
                    n2 = np.where(n2)[0][0]
                    self.srcSurfAmp[i] = np.prod(2/(1+b2[n2,:]))
                elif np.any(n3):
                    n3 = np.where(n3)[0][0]
                    self.srcSurfAmp[i] = np.prod(2/(1+b3[n3,:]))
                # Source strength is amplitude and surface factor combined
                self.srcStrength[i] = self.srcAmp[i] * self.srcSurfAmp[i]
        
        # Receivers
        for i in range(self.recN):
            # Flat index
            self.recFlatInd[i] = np.ravel_multi_index(self.recInd[i], self.Nxyz)
            # Find node type
            self.recNodeType[i] = nodes.flat[self.recFlatInd[i]]
            # 'Strength'
            self.recStrength[i] = self.recAmp[i]
    
    def abs2Admit(self, alpha):
        
        # Convert normal incidence absorption coefficient data to normalised admittance
        R = np.sqrt(1-alpha)
        admit = (1-R)/(1+R)
        return admit
        
    def srcOnMesh(self):
        
        # Check valid source positions
        
        # Update on debug
        self.printToDebug('srcOnMesh')
            
        onMesh = False
        for i, nodeType in enumerate(self.srcNodeType):
            if self.srcAmp[i]!=0 and nodeType != 0:
                onMesh = True
        return onMesh
    
    def recOnMesh(self):
        
        # Check valid receiver positions
        
        # Update on debug
        self.printToDebug('recOnMesh')
        
        onMesh = False
        for i, nodeType in enumerate(self.recNodeType):
            if self.recAmp[i]!=0 and nodeType != 0:
                onMesh = True
        return onMesh
    
    def srcRecOnMesh(self):
        
        # Check valid source and receiver positions
        
        onMesh = self.srcOnMesh() or self.recOnMesh()
        return onMesh
        
    def indFixDims(self, inds, Ni, dims):
        
        # Shift dimensions of flat indices
        
        # Number of dimensions
        NDims = dims.size
        # Number of indices
        NInd = inds.size
        if NInd > 0:
            # Unflatten indices (as numpy array)
            inds = np.unravel_index(inds, Ni)
            inds = np.array(inds)
            # Shift dimensions
            invDims = [np.where(i==dims)[0][0] for i in range(NDims)]
            Ni = Ni[invDims]
            inds = inds[invDims,:]
            # Convert back to flat indices
            inds = np.ravel_multi_index(inds, Ni)
        
        return inds
    
    def newEmptyMesh(self, Nxyz=NXYZ):
        
        # define new empty mesh size
        
        # Update on debug
        self.printToDebug('newEmptyMesh')
        
        # Set the new size
        self.Nxyz = Nxyz
        # Set new inputs
        self.setInputs()
        # Define using makeReset
        self.meshReset()
    
    def meshReset(self, doPrepareMesh=False):
        
        # Reset to blank mesh
        
        # Update on debug
        self.printToDebug('meshReset')
            
        #self.mesh = np.zeros((self.Nxyz), dtype=int)
        self.mesh = np.zeros((self.Nxyz))
        if doPrepareMesh:
            self.prepareMesh()
        
    def image2Mesh(self, threshold=None, addToPlot=True, 
                   xMesh=None, yMesh=None, xOffset=0, yOffset=0):
        
        # Make binary (black and white) or greyscale version of mesh from image
        # 2D mesh is assumed
        
        # Update on debug
        self.printToDebug('image2Mesh')
        
        if threshold != None:
            self.imageThreshold = threshold
        
        # Get mesh from image
        
        if self.image is None:
            
            # Size of requested mesh
            if xMesh is None:   xMesh=self.Nxyz[0]
            if yMesh is None:   yMesh=self.Nxyz[1]
            
            # Make empty mesh if no image
            self.mesh = np.zeros((xMesh, yMesh))
            
        else:
            
            # Otherwise...
            
            # Size of image
            imX, imY = self.image.shape
            # Size of requested mesh
            if xMesh is None:   xMesh=imX
            if yMesh is None:   yMesh=imY
            
            # Opposite of image (white = 0, black = 255)
            self.mesh = 255-self.image
            
            # Make sure floats
            self.mesh = self.mesh.astype(float)
            
            # Set to zero if below threshold
            self.mesh[self.mesh <= 255*self.imageThreshold] = 0
            
            # # Scale if doing 'varying' or otherwise set rest to one
            # if self.betaMode == 'varying':
            #     self.mesh = self.mesh * 1/255
            # else:
            #     self.mesh[self.mesh > 255*self.imageThreshold] = 1
            
            # Now always do scaling so image doesn't get made binary
            self.mesh = self.mesh * 1/255
            
            # Pad/trim if not same size
            if xMesh!=imX or yMesh!=imY:
                # Size of new mesh
                if xMesh is None: xMesh = imX
                if yMesh is None: yMesh = imY
                # Start and end indices for new mesh
                x1 = max(xOffset,0)
                y1 = max(yOffset,0)
                x2 = min(imX+xOffset,xMesh)
                y2 = min(imY+yOffset,yMesh)
                # Start and end indices for original mesh
                x10 = -min(xOffset,0)
                y10 = -min(yOffset,0)
                x20 = min(xMesh-xOffset,imX)
                y20 = min(yMesh-yOffset,imY)
                # Make new blank mesh and add pad/trimmed orig
                meshOrig = self.mesh
                self.mesh = np.zeros((xMesh,yMesh))
                self.mesh[x1:x2,y1:y2] = meshOrig[x10:x20,y10:y20]
        
        # Set sim to match new mesh size
        self.Nxyz = list(self.mesh.shape)
        
        # Add to plot 
        if addToPlot:
            self.updatePlotMask()
    
    def repDecMesh(self, rep=None, dec=None):
        
        # Repeat and/or decimate mesh
        if rep != None:
            self.mesh = self.repeatData(self.mesh, rep)
        if dec != None:
            self.mesh = self.decimateData(self.mesh, dec)
        self.Nxyz = list(self.mesh.shape)
    
    def updateWithImage(self):
        
        # Make new mesh array but don't plot
        self.image2Mesh(addToPlot=False)
        # Reset inputs
        self.setInputs()
        # Reset pressures etc..
        self.runReset()
        # Add mesh array to plot
        self.updatePlotMask()
        
    def loadImage(self, file, doReset=True, sizeLimits = None):
        
        # Update on debug
        self.printToDebug('loadImage')
        
        # Load image from file (as normalised greyscale)
        self.image = Image.open(file).convert('L')
        # Convert to numpy array and flip to account for direction in 
        # 'vertical' data
        self.image = np.asarray(self.image)
        self.image = np.flip(self.image, axis=0)
        # Clear mesh
        self.mesh = None
        
        if not sizeLimits is None:
            # If size limits then check if needs trimming/padding
            self.image, isAltered = self.padOrTrim(self.image, sizeLimits, 255)
        else:
            isAltered = False
        
        # Reset everything to take in to account new grid size
        if doReset:
           self.updateWithImage()
           
        return isAltered
    
    def padOrTrim(self, arr, sizeLims, padValue):
        
        # Pad or trim array based on provided size limits
        isAltered = False
        
        # Size of array
        ND = len(arr.shape)
        for i, arrSize in enumerate(arr.shape):
            if arrSize < sizeLims[2*i]:
                # Pad with zeros
                isAltered = True
                NPad = sizeLims[2*i]-arrSize
                pad = np.zeros((ND,2), dtype=int)
                pad[i,1] = NPad
                arr = np.pad(arr, pad, 'constant', constant_values=padValue)
            elif arrSize >= sizeLims[2*i+1]:
                # Trim
                isAltered = True
                NTrim = sizeLims[2*i+1]
                arr = np.delete(arr,np.s_[NTrim:],i)
        
        return arr, isAltered
    
    def getInds(self, xyz):
        
        # Get indices for location
        inds = [round(x/self.X) for x in xyz]
        # # Because first dimension other way around
        # inds[0] = self.Nxyz[0]-inds[0]-1
        return inds
        
    def addSrc(self, xyz, tOffset=0.0, f0=None, srcType=None):
        
        # Update on debug
        self.printToDebug('addSrc')
        
        # Add a source
        if self.checkCoords(xyz):
            self.srcXyz.insert(self.srcN, xyz)
            inds = self.getInds(xyz)
            self.srcInd.insert(self.srcN, inds)
            self.srcFlatInd.insert(self.srcN, \
                                   np.ravel_multi_index(inds, self.Nxyz))
            self.srcXyzDisc.insert(self.srcN, \
                                   [ind*self.X for ind in self.srcInd[self.srcN]])
            self.srcType.insert(self.srcN, srcType)
            srcData, tOffset, f0, srcType = self.getSrc(tOffset=tOffset,
                                                        f0=f0, 
                                                        srcType=srcType, 
                                                        i=self.srcN)
            self.srcData.insert(self.srcN, srcData)
            self.srcT0.insert(self.srcN, tOffset)
            self.srcFreq.insert(self.srcN, f0)
            self.srcAmp.insert(self.srcN, 1.0)
            self.srcType.insert(self.srcN, srcType)
            self.srcStrength.insert(self.srcN, 1.0)
            self.srcNodeType.insert(self.srcN, 0)
            self.srcN += 1
        
    def moveSrc(self, xyz, i=-1):
        
        # Update on debug
        self.printToDebug('moveSrc')
        
        # Move source to position
        validMove = self.checkCoords(xyz)
        if validMove:
            self.srcXyz[i] = xyz
            self.srcInd[i] = self.getInds(xyz)
            self.srcFlatInd[i] = np.ravel_multi_index(self.srcInd[i], self.Nxyz)
            self.srcXyzDisc[i] = [ind*self.X for ind in self.srcInd[i]]
        
        return validMove
    
    def delSrc(self, i=-1):
        
        # Update on debug
        self.printToDebug('delSrc')
            
        # Delete a source
        self.srcXyz.pop(i)
        self.srcInd.pop(i)
        self.srcFlatInd.pop(i)
        self.srcXyzDisc.pop(i)
        self.srcT0.pop(i)
        self.srcFreq.pop(i)
        self.srcData.pop(i)
        self.srcAmp.pop(i)
        self.srcType.pop(i)
        self.srcStrength.pop(i)
        self.srcNodeType.pop(i)
        self.srcN -= 1
        
    def addRec(self, xyz):
        
        # Update on debug
        self.printToDebug('addRec')
            
        # Add a receiver
        if self.checkCoords(xyz):
            self.recXyz.insert(self.recN, xyz)
            inds = self.getInds(xyz)
            self.recInd.insert(self.recN, inds)
            self.recFlatInd.insert(self.recN, \
                                   np.ravel_multi_index(inds, self.Nxyz))
            self.recXyzDisc.insert(self.recN, \
                                   [ind*self.X for ind in self.recInd[self.recN]])
            self.recData.insert(self.recN, np.zeros(self.Nt))
            self.recAmp.insert(self.srcN, 1.0)
            self.recStrength.insert(self.srcN, 1.0)
            self.recNodeType.insert(self.recN, 0)
            self.recN += 1
    
    def moveRec(self, xyz, i=-1):
        
        # Update on debug
        self.printToDebug('moveRec')
            
        # Move source to position
        validMove = self.checkCoords(xyz)
        if validMove:
            self.recXyz[i] = xyz
            self.recInd[i] = self.getInds(xyz)
            self.recFlatInd[i] = np.ravel_multi_index(self.recInd[i], self.Nxyz)
            self.recXyzDisc[i] = [ind*self.X for ind in self.recInd[i]]
        return validMove
    
    def delRec(self, i=-1):
        
        # Update on debug
        self.printToDebug('delRec')
            
        # Delete a receiver
        self.recXyz.pop(i)
        self.recInd.pop(i)
        self.recFlatInd.pop(i)
        self.recXyzDisc.pop(i)
        self.recData.pop(i)
        self.recAmp.pop(i)
        self.recStrength.pop(i)
        self.recNodeType.pop(i)
        self.recN -= 1
        
    def getSrc(self, tOffset=0.0, f0=None, srcType=None, i=-1):
        
        # Centre frequency
        if f0 is None:
            # Default to half the standard dispersion limit
            f0 = self.fs*0.075/2
        
        # Source type
        if srcType is None:
            # Default source type
            srcType = SRC_TYPE_DEFAULT
        srcTypeList = srcType.lower().split()
        
        # Time vector
        t = np.arange(0,self.Nt,1)/self.fs
        t -= tOffset
        
        if srcTypeList[0][0] == "u":            # User defined
            src_fcn = self.srcData[i]
        
        elif srcTypeList[0][0] == "i":          # Impulse
            # src_fcn = np.zeros(self.Nt)
            # src_fcn[0] = 1.0
            src_fcn = np.sinc(t*self.fs)
        
        elif srcTypeList[0][0] == "g":          # Gaussian
            if len(srcTypeList) > 1:
                if srcTypeList[1][0] == "d":    # Derivative
                    # Derivative order
                    if len(srcTypeList) > 2:
                        derivNum = int(srcTypeList[2])
                    else:
                        derivNum = 1
                    # Get derivative
                    if derivNum == 1:
                        # 1st order Gaussian derivative, with appropriate offset, unity 
                        # scaling, and target peak frequency in frequency domain
                        alpha = 2*(PI*f0)**2
                        t0 = 4*np.sqrt(1/alpha)
                        dt = t-t0
                        src_fcn = (-np.sqrt(2*alpha)*np.exp(0.5)*dt)*np.exp(-alpha*dt**2)
                    elif derivNum == 2:
                        # As above, but 2nd order
                        alpha = (PI*f0)**2
                        t0 = 4*np.sqrt(1/alpha)
                        dt = t-t0
                        src_fcn = (1-2*alpha*dt**2)*np.exp(-alpha*dt**2)
                    elif derivNum == 3:
                        # As above, but 3rd order
                        alpha = (2/3)*(PI*f0)**2
                        t0 = 4*np.sqrt(1/alpha)
                        dt = t-t0
                        src_fcn = -dt*(2*alpha*dt**2-3)*np.exp(-alpha*dt**2)
                        src_peak = np.sqrt((3-np.sqrt(6))/(2*alpha))* \
                            np.sqrt(6)*np.exp(-(3-np.sqrt(6))/2)
                        src_fcn /= src_peak
            else:
                # Gaussian pulse with -3 dB point at f0
                alpha = (PI*f0)**2/(-np.log(1/np.sqrt(2)))
                t0 = 3*np.sqrt(1/alpha)
                dt = t-t0
                src_fcn = np.exp(-alpha*dt**2)
        
        elif srcTypeList[0][0] == "t":          # Tone
            if len(srcTypeList) > 1:
                if srcTypeList[1][0] == "p":    # Pulse
                    # Pulse cycles
                    if len(srcTypeList) > 2:
                        P = int(srcTypeList[2])
                    else:
                        P = 1
                    # Get tone pulse
                    src_fcn = np.cos(2*PI*f0*t-PI)
                    env = 0.5*(np.cos(2*PI*(f0/P)*t-PI)+1)
                    env = np.where(t<0, 0, env)
                    env = np.where(t>P/f0, 0, env)
                    src_fcn *= env
                    src_fcn *= (-1)**(P-1)
            else:
                # Constant tone
                src_fcn = np.sin(2*PI*f0*t)
        
        return src_fcn, tOffset, f0, srcType
    
    def srcRecDataReset(self):
        
        # Reset all src/rec data
        for i in range(0,self.srcN):
            self.srcData[i], self.srcT0[i], self.srcFreq[i], self.srcType[i] =\
                self.getSrc(tOffset=self.srcT0[i], f0=self.srcFreq[i], \
                            srcType=self.srcType[i], i=i)
        for i in range(0,self.recN):
            self.recData[i] = np.zeros(self.Nt)
    
    def writeRecData(self):
        
        # Write receiver data to file
        
        # Update on debug
        self.printToDebug('writeRecData')
        
        # Write to file
        if self.recN > 0:
            # Reshape output data
            data = np.transpose(np.array(self.recData))
            # Keep data where amplplitude not zero
            data = data[:,np.array(self.recAmp)!=0]
            # Sample rate to use
            if self.fsOut is None:
                wavFs = self.fs
            else:
                wavFs = self.fsOut
                data = self.resampleData(data, wavFs, self.fs)
            # Write to file
            if data.shape[1] > 0:
                wavfile.write(self.recDataFile, wavFs, data)
    
    def writeGifData(self):
        
        # Write GIF data to file
        
        # Update on debug
        self.printToDebug('writeGifData')
        
        self.imgs[0].save(self.gifFile, format='gif', \
                            save_all=True, append_images=self.imgs[1:], \
                            duration=self.gifFrameTime, 
                            loop=self.gifLoopNum)
    
    def resampleData(self, data, newFs, oldFs):
        
        # Resample data
        
        # Ratio of output to input sample rate
        fsRatio = newFs/oldFs
        # Number of channels
        NChan = data.shape[1]
        # Input and desired output size
        NIn = data.shape[0]
        NOut = np.ceil(NIn*fsRatio)
        # Round up to nearest full second
        N = (np.ceil(NIn/oldFs)*oldFs).astype(int)
        NNew = np.ceil(N*fsRatio).astype(int)
        # Pad
        padData = np.zeros((N,NChan))
        padData[0:NIn,:] = data
        # Resample
        data = resample(padData, NNew)
        # Trim
        NOut = max(NOut,NNew)
        data = data[0:NOut,:]
        
        return data
    
    def repeatData(self, data, rep):
        
        # Repeat data in each dimension
        for i, r in enumerate(rep):
            r = int(max(1, r))
            data = data.repeat(r, axis=i)
        return data
    
    def decimateData(self, data, dec):
        
        # Reduce data using decimation
        for i, d in enumerate(dec):
            d = int(min(1, d))
            data = np.delete(data,np.s_[0::d],i)
        return data
    
    def setCMap(self):
        
        # Make colour map
        cDict = {
            'red': (
                (0.0,   self.c1[0], self.c1[0]),
                (0.5,   self.c0[0], self.c0[0]),
                (1.0,   self.c2[0], self.c2[0]),
            ),
            'green': (
                (0.0,   self.c1[1], self.c1[1]),
                (0.5,   self.c0[1], self.c0[1]),
                (1.0,   self.c2[1], self.c2[1]),
            ),
            'blue': (
                (0.0,   self.c1[2], self.c1[2]),
                (0.5,   self.c0[2], self.c0[2]),
                (1.0,   self.c2[2], self.c2[2]),
            )
        }
        self.cMap = LinearSegmentedColormap('fdtdCustom', cDict)
        
        if self.doPlot and USE_MATPLOTLIB:
            if self.plotExist():
                self.hPlot.set_cmap(self.cMap)
    
    def arr2CMap(self, img, cLims, cMap):
        
        # Convert to correct format using colour limits and map
        cRange = cLims[1]-cLims[0]
        img = (img-cLims[0])*1/cRange           # Scale
        img = np.clip(img,0,1)                  # Clip
        img = cMap(img)                         # Apply colour map
        img *= 255                              # Scale to 255
        img = img.astype(np.uint8)              # As uint8 data
        img = np.flipud(img)                    # Flip to make right way up
        return img
    
    def updateGif(self):
        
        # Add frame to GIF
        
        # If first frame
        frame0 = len(self.imgs)==0
        
        # Add to GIF (if doing)
        if self.saveGif and (self.updatePlotThisLoop or frame0):
            
            # If first frame
            if frame0:
                
                # If GIF area defined
                if self.gifArea != None:
                    # GIf area no bigger than current mesh
                    self.gifArea[0] = int(max(0, self.gifArea[0]))
                    self.gifArea[1] = int(max(0, self.gifArea[1]))
                    self.gifArea[2] = int(min(self.Nxyz[0], self.gifArea[2]))
                    self.gifArea[3] = int(min(self.Nxyz[1], self.gifArea[3]))
                    # If doing trimming of GIF
                    self.gifTrim =  self.gifArea[0] > 0 or \
                                    self.gifArea[1] > 0 or \
                                    self.gifArea[2] < self.Nxyz[0] or \
                                    self.gifArea[3] < self.Nxyz[1]
                else:
                    self.gifTrim = False
            
                # If there is a mask (surfaces) to get
                if self.plotShowMask:
                    if self.NDim > 1:
                        # Convert to correct format using colour map
                        self.imgMesh = self.arr2CMap(self.meshSlice, \
                                                     [0.0, 1.0], self.gMap)
                    elif self.NDim == 1:
                        # TODO - What to plot
                        self.imgMesh = np.zeros((10,10)) # TEMP BLANK!!!!!
                    else:
                        # Invalid NDim
                        ()
            
            # Get image
            if self.NDim == 3:
                img = self.p[:,:,self.plotSliceNum]
            elif self.NDim == 2:
                img = self.p
            elif self.NDim ==1:
                # TODO - What to plot
                img = np.zeros((10,10)) # TEMP BLANK!!!!!
            else:
                # Invalid NDim
                ()
            
            # Convert to correct format using colour map
            img = self.arr2CMap(img, self.cLims, self.cMap)
            
            # If there is a mask (surfaces) to add then combine
            # (Getting confused about why the need for flipud!!)
            if self.plotShowMask:
                for i in range(0,3):
                    img[:,:,i] = np.where(np.flipud(self.mesh)>0, \
                                          self.imgMesh[:,:,i], \
                                              img[:,:,i])
            
            # Trim if requested
            if self.gifTrim:
                img = img[self.gifArea[0]:self.gifArea[2], \
                          self.gifArea[1]:self.gifArea[3],:]
            
            # Add to list
            self.imgs.append(Image.fromarray(img).convert('RGB'))
        
    def getMeshSlice(self):
        
        # Get mesh slice for plottng
        if self.mesh is None or not self.plotShowMask:
            self.meshSlice = np.zeros(self.Nxyz, dtype=np.uint8)
        elif self.NDim == 1:
            self.meshSlice = self.mesh.reshape([1,len(self.mesh)])
        elif self.NDim == 2:
            self.meshSlice = self.mesh
        elif self.NDim == 3:
            self.meshSlice = self.mesh[:,:,self.plotSliceNum]
        
        # Add mask
        self.meshSlice = np.ma.masked_where(self.meshSlice==0, self.meshSlice)
        
    def makePlot(self):
        
        # Make plot
        # Note: extent is specified other way around
        
        # Update on debug
        self.printToDebug('makePlot')
        
        # Slice to plot if 3D
        try: self.plotSliceNum = self.srcInd[0][-1]
        except: self.plotSliceNum = 0
        
        # If plotting mesh
        if self.plotShowMask and \
            ( (self.doPlot and USE_MATPLOTLIB) or \
            (self.saveGif) ):
            # Mask
            self.getMeshSlice()
            # Colour map
            gMap = 'gray'
            if not self.plotMaskColInvert:
                gMap += '_r'
            self.gMap = plt.get_cmap(gMap)
        
        if self.doPlot and USE_MATPLOTLIB:
            
            # Make or clear
            if self.plotExist():
                self.ax.cla()
                if self.NDim > 1:
                    # Lazy way to not check if colBar exists
                    try: self.colBar.remove()
                    except: ()
            else:
                #plt.ion()
                self.hf, self.ax = plt.subplots()
                self.figNum = self.hf.number
                
            # Plot range
            if self.NDim == 1:
                extent = [-self.X*0.5, self.D[0]+self.X*0.5]+self.cLims
                aspect = 'auto'
            else:
                extent = [-self.X*0.5, self.D[1]+self.X*0.5, \
                          -self.X*0.5, self.D[0]+self.X*0.5]
                aspect = 'equal'
            
            # Main p plot
            if self.NDim == 3:
                # Plot slice
                self.hPlot = self.ax.imshow(self.p[:,:,self.plotSliceNum], \
                   extent=extent, \
                   vmin=self.cLims[0], vmax=self.cLims[1], \
                   cmap=self.cMap, \
                   aspect='equal', \
                   origin='lower')
                self.ax.set_xlim([0, self.D[1]])
                self.ax.set_ylim([0, self.D[0]])
            elif self.NDim == 2:
                # Plot all
                self.hPlot = self.ax.imshow(self.p, \
                   extent=[0.0, self.D[1], 0.0, self.D[0]], \
                   vmin=self.cLims[0], vmax=self.cLims[1], \
                   cmap=self.cMap, \
                   aspect='equal', \
                   origin='lower')
                self.ax.set_xlim([0, self.D[1]])
                self.ax.set_ylim([0, self.D[0]])
            elif self.NDim == 1:
                xx = np.linspace(0.0, self.D[0], self.Nxyz[0], endpoint=True)
                self.hPlot, = self.ax.plot(xx,self.p)   # Note comma as returns tuple
                self.ax.grid()
                self.ax.set_xlim([0, self.D[0]])
                self.ax.set_ylim(self.cLims)
            else:
                # Invalid NDim
                ()
            
            # TODO - Why not working if do this??
            # self.hPlot = self.ax.imshow(self.p)
            # self.hPlot.set_extent=[0.0, self.D[1], 0.0, self.D[0]]
            # self.hPlot.set_vmin=self.cLims[0]
            # self.hPlot.set_vmax=self.cLims[1]
            # self.hPlot.set_cmap=self.cMap
            # self.hPlot.set_aspect='equal'
            # self.hPlot.set_origin='lower'
            
            # If plotting colour bar
            if self.plotShowColBar:
                if self.NDim > 1:
                    self.colBar = plt.colorbar(self.hPlot)
            
            # If plotting mesh
            if self.plotShowMask:
                # Plot
                self.hPlotMask = self.ax.imshow(self.meshSlice, \
                    extent=extent, \
                    vmin=0, vmax=1, \
                    cmap=self.gMap, \
                    interpolation='none', \
                    aspect=aspect, \
                    origin='lower')
            
            # If plotting sources
            if self.plotShowSrc:
                if self.NDim == 1:
                    xx = np.array(self.srcXyzDisc)[:,0]
                    yy = np.zeros(len(xx))
                    onSlice = np.full(len(xx),True)
                else:
                    xx = np.array(self.srcXyzDisc)[:,1]
                    yy = np.array(self.srcXyzDisc)[:,0]
                    if self.NDim == 2:
                        onSlice = np.full(len(xx),True)
                    else:
                        zInd = np.array(self.srcInd)[:,2]
                        onSlice = self.plotSliceNum==zInd
                # Colours
                col1 = (0.3,0.3,0.4,1)
                col2 = (0.8,0.8,0.8,1)
                cols = [col1]*len(xx)
                for i in range(len(xx)):
                    if not onSlice[i]:
                        cols[i] = col2
                # Plot
                self.hPlotSrc = self.ax.scatter(xx, yy, c=cols, marker="o")
            
            # If plotting receivers
            if self.plotShowRec:
                if self.NDim == 1:
                    xx = np.array(self.recXyzDisc)[:,0]
                    yy = np.zeros(len(xx))
                    onSlice = np.full(len(xx),True)
                else:
                    xx = np.array(self.recXyzDisc)[:,1]
                    yy = np.array(self.recXyzDisc)[:,0]
                    if self.NDim == 2:
                        onSlice = np.full(len(xx),True)
                    else:
                        zInd = np.array(self.recInd)[:,2]
                        onSlice = self.plotSliceNum==zInd
                # Colours
                col1 = (0.4,0.3,0.3,1)
                col2 = (0.8,0.8,0.8,1)
                cols = [col1]*len(xx)
                for i in range(len(xx)):
                    if not onSlice[i]:
                        cols[i] = col2
                # Plot
                self.hPlotRec = self.ax.scatter(xx, yy, c=cols, marker="x")
                
            # Draw
            self.hf.show()
            #plt.show()
        
        # Update GIF - first frame (if doing)
        self.imgs = []
        self.updateGif()
            
    def setPlotParameters(self):
        
        # UNUSED
        
        # Set parameters of plot
        if self.doPlot and USE_MATPLOTLIB:
            if self.plotExist():
                ()
        
    def updatePlot(self, forceUpdate=False):
            
        # Update plot
        if self.doPlot and USE_MATPLOTLIB:
            if self.plotExist():
                if self.updatePlotThisLoop or forceUpdate:
                    # Update data based on no. dimensions
                    if self.NDim == 3:
                        self.hPlot.set_data(self.p[:,:,self.plotSliceNum])
                    elif self.NDim == 2:
                        self.hPlot.set_data(self.p)
                    elif self.NDim == 1:
                        self.hPlot.set_ydata(self.p)
                    else:
                        # Invalid NDim
                        ()
                    # Draw
                    #plt.draw()
                    #plt.pause(self.plotUpdateTime)
                    self.hf.canvas.draw()
                    self.hf.canvas.flush_events()
                    # time.sleep(self.plotUpdateTime)                    
            else:
                self.stop()
        
        # Save GIF (if doing)
        self.updateGif()
                
    def updatePlotMask(self):
        
        # Update plot mask
        if self.doPlot and USE_MATPLOTLIB and self.plotShowMask:
            if self.plotExist():
                self.getMeshSlice()
                self.hPlotMask.set_data(self.meshSlice)
                self.hf.canvas.draw()
                self.hf.canvas.flush_events()
        
    def plotExist(self):
        
        # Check if plot exists
        hfExist = False
        if self.doPlot and USE_MATPLOTLIB:
            if not self.figNum is None:
                if plt.fignum_exists(self.figNum):
                    hfExist = True
        
        return hfExist
    
    def printToDebug(self, txt):
        
        # Print debug message
        if self.debug:
            print(self.debugPrefix + str(txt))
