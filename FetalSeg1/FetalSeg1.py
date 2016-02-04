#% AUTOMATIC FEMUR SEGMENTATION AND LENGTH MEASUREMENT FROM FETAL ULTRASOUND 
#% IMAGES
#%
#% Developer:    Nikhil Narayan S
#%
#% Date:         05/02/2016
#% 
#% Description: This file contains the implementation of the algorithm 
#%              proposed in [1] as a part of the ISBI-2012 challenge. 
#%              It was shown in [2] that this algorithm outperformed
#%              other methods that were submitted for the challenge to
#%              measure the length of Femur .
#%
#%              [1]. Wang, Ching-Wei, et al. "Automatic femur segmentation 
#%                   and length measurement from fetal ultrasound images." 
#%                   Proceedings of Challenge US: Biometric Measurements 
#%                   from Fetal Ultrasound Images, ISBI 2012 (2012): 21-23.                  
#%
#%              [2]. Rueda, Sylvia, et al. "Evaluation and comparison of 
#%                   current fetal ultrasound image segmentation methods 
#%                   for biometric measurements: a grand challenge." 
#%                   IEEE Transactions on Medical Imaging, 33.4 (2014): 
#%                   797-813.
#%**************************************************************************

import Tkinter as tk
from tkFileDialog import askopenfilename
from skimage import io, filters, exposure
from skimage.morphology import disk, dilation, remove_small_objects
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

root = tk.Tk()
root.withdraw()
filename = askopenfilename()
print(filename)

pInputImage = io.imread(filename,as_grey=True)

pOriginalImage = pInputImage

#-----------------------------------------------------------------
# Apply median filter to remove speckle noise
#-----------------------------------------------------------------
pMedFiltImg = filters.median(pInputImage,disk(5))

pTotalPix = pInputImage.size

#-----------------------------------------------------------------
# Compute the histogram
#-----------------------------------------------------------------
pHist = np.array(exposure.histogram(pMedFiltImg,nbins=256)).astype(float)

pProb = (pHist[0] / pTotalPix) + np.spacing(1)

pHProb = -pProb * np.log10(pProb)

H = np.empty([1,255],dtype=float)

#-----------------------------------------------------------------
# Perform entropy based segmentation
#-----------------------------------------------------------------

for i in xrange(0,255,1):
    HA = sum(pHProb[0:i])
    HB = sum(pHProb[i+1:256])
    PA = sum(pProb[0:i])+ np.spacing(1)
    PB = sum(pProb[i+1:256]) + np.spacing(1)
    H[0,i] = -np.log10(PA)-np.log10(PB)-(HA/(PA))-(HB/(PB))

nThre = H.argmax(1)

pInputImage[pInputImage<nThre] = 0
pInputImage[pInputImage>=nThre] = 255

#-----------------------------------------------------------------
# Apply morphological operators to segment the Femur
#-----------------------------------------------------------------
pInputImage = dilation(pInputImage,disk(2))

[pInputImage,nLabel] = label(pInputImage,connectivity=2,return_num=True)

pInputImage = remove_small_objects(pInputImage,min_size=300,in_place=True)

pInputImage[pInputImage>0] = 255

[pInputImage,nLabel] = label(pInputImage,connectivity=1,return_num=True)

props = regionprops(pInputImage)

pRatio = [abs(float(props[i].bbox[2])-float(props[i].bbox[0]))/abs(float(props[i].bbox[3])-float(props[i].bbox[1])) for i in xrange(0,nLabel-1)]

nSleekLabel = np.argmin(pRatio)

pInputImage[pInputImage!=nSleekLabel+1] = 0
pInputImage[pInputImage>0] = 255

nLength = np.sqrt((float(props[nSleekLabel].bbox[2])-float(props[nSleekLabel].bbox[0]))**2+(float(props[nSleekLabel].bbox[3])-float(props[nSleekLabel].bbox[1]))**2)
#nLength = np.sqrt((float(props[inSleekLabel].bbox[2])-float(props[nSleekLabel].bbox[0]))**2+(float(props[nSleekLabel].bbox[3])-float(props[nSleekLabel].bbox[1])**2)
#-----------------------------------------------------------------
# Display segmented image
#-----------------------------------------------------------------
print "Length of Femur = ", nLength, "pixels"
#io.imshow(pInputImage)
#io.show()