from skimage import io, filters, exposure
from skimage.morphology import disk
import numpy as np

pInputImage = io.imread("F:\\PhD_Data\\Normal\\BMP\\TMIPMP\\All\\M_20120209171718.bmp")

pMedFiltImg = filters.median(pInputImage,disk(5))

pTotalPix = pInputImage.size

pHist = np.array(exposure.histogram(pMedFiltImg,nbins=256))
pHist = pHist.astype(float)
pProb = pHist[0] / pTotalPix

print pProb
#io.imshow(pInputImage)
#io.show()
#io.imshow(pMedFiltImg)
#io.show()