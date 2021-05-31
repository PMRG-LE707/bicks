import numpy as np
from PhotonicCrystalSlab import PhotonicCrystalSlab, EssentialNumber
from Field import testCTIR, FieldsWithCTIRSingle
import matplotlib.pyplot as plt

qorky = [0.1181, 0.1713, 0.072, 0.1311,
         0.3156, 0.212, 0.044, 0.2782,
         0.233, 0.41, 0.206, 0.1781,
         0.28, 0.426]
k0 = [0.75398, 0.7632, 0.80759, 0.76285,
      0.46092, 0.60472, 0.636, 0.924,
      1.306, 1.62, 0.5687, 0.59063,
      0.48, 0.428]
hset = [1.4, 2, 2.5, 3,
        1.4, 1.45, 1.375, 15.8,
        1.2, 12, 2.1, 1.97,
        1.315789, 1.105]

BICky = [0.053, 0.044, 0, 0,
         0, 0.212]
BICk0 = [0.597, 0.636, 0.64341, 0.49824,
         0.58665, 0.60472]
BICh = [1.4635, 1.375, 1.2897, 2,
        2, 1.45]
BICq = [0, 0, 0, 0.2264,
        0.1841, 0]
nnu = -1
# properties of PhC slab
a = 1
h = BICh[nnu] * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(ep, fr, a,thickness=h)
k0a = 0.49824 * 2 * np.pi
qa = 0.212 * 2 * np.pi
kya = 0.05  * 2 * np.pi
realdata=[]
imagdata=[]
number=40
for i in range(number):
    num = EssentialNumber(n_radiation=1,nimag_plus=i)
   
    data =  testCTIR(phcs, num, k0a, qa, kya).coefs_inside
    realdata.append(data.real)
    imagdata.append(data.imag)
xdata=range(number)
plt.scatter(xdata,realdata) 
plt.scatter(xdata,imagdata) 
"""
cTIRfield = FieldsWithCTIRSingle(phcs, num, k0a, qa, kya, mode="mix")
print(cTIRfield.even_coefs_inside)
print(cTIRfield.odd_coefs_inside)
"""
