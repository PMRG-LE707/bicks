import numpy as np
from PhotonicCrystalSlab import PhotonicCrystalSlab, EssentialNumber
from Field import FieldsWithCTIR, FieldsWithCTIRSingle


qorky = [0.1181, 0.1713, 0.072, 0.1311, 0.3156, 0.212, 0.044]
k0 = [0.75398, 0.7632, 0.80759, 0.76285, 0.46092, 0.60472, 0.636]
hset = [1.4, 2, 2.5, 3, 1.4, 1.45, 1.375]
nnu = -3
# properties of PhC slab
a = 1
h = hset[nnu] * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(h, ep, fr, a)
num = EssentialNumber()

k0a = k0[nnu] * 2 * np.pi
qa = qorky[nnu] * 2 * np.pi
kya = 0  * 2 * np.pi
cTIRfield = FieldsWithCTIRSingle(phcs, num, k0a, qa, kya)
print(cTIRfield.odd_coefs_inside)



