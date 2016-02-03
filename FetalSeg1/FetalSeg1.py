from skimage import io, filters, exposure
from skimage.morphology import disk
import numpy as np

pInputImage = io.imread("F:\\ISBI_Challenge\\FetalSeg1\\femur-01.jpg")

pMedFiltImg = filters.median(pInputImage,disk(5))

pTotalPix = pInputImage.size

pHist = np.array(exposure.histogram(pMedFiltImg,nbins=256)).astype(float)

pProb = (pHist[0] / pTotalPix) + np.spacing(1)

pHProb = -pProb * np.log10(pProb)

H = np.empty([1,255],dtype=float)
for i in xrange(0,255,1):
    HA = sum(pHProb[0:i])
    HB = sum(pHProb[i+1:256])
    PA = sum(pProb[0:i])+ np.spacing(1)
    PB = sum(pProb[i+1:256]) + np.spacing(1)
    H[0,i] = -np.log10(PA)-np.log10(PB)-(HA/(PA))-(HB/(PB))

nThre = H.argmax(1)

pInputImage[pInputImage<nThre] = 0
pInputImage[pInputImage>=nThre] = 255
io.imshow(pInputImage)
io.show()