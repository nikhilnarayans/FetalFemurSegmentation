from skimage import io, filters, exposure
from skimage.morphology import disk, dilation, remove_small_objects
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

pInputImage = io.imread("F:\\ISBI_Challenge\\FetalSeg1\\femur-01.jpg",as_grey=True)

pOriginalImage = pInputImage

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

pImages = io.imshow(pInputImage)
io.show()