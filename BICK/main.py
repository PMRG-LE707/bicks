import numpy as np
from PhotonicCrystalSlab import PhotonicCrystalSlab, EssentialNumber
from Field import FieldsWithCTIR, FieldsWithCTIRSingle


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
BICky = [0.053, 0.044, 0]
BICk0 = [0.597, 0.636, 0.64341]
BICh = [1.4635, 1.375, 1.2897]
BICq = [0, 0, 0]
nnu = 0
# properties of PhC slab
a = 1
h = BICh[nnu] * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(ep, fr, a,thickness=h)
num = EssentialNumber(n_radiation=1,nimag_plus=1)

k0a = BICk0[nnu] * 2 * np.pi
qa = BICq[nnu] * 2 * np.pi
kya = BICky[nnu]  * 2 * np.pi
cTIRfield = FieldsWithCTIRSingle(phcs, num, k0a, qa, kya, mode="mix")
print(cTIRfield.even_coefs_inside)
print(cTIRfield.odd_coefs_inside)
