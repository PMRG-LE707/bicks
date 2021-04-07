import numpy as np
from PhotonicCrystalSlab import PhotonicCrystalSlab
from Field import FieldsWithCTIR
import time

q = [0.1181, 0.1713, 0.072, 0.1311, 0.3156, 0.212]
k0 = [0.75398, 0.7632, 0.80759, 0.76285, 0.46092, 0.60472]
hset = [1.4, 2, 2.5, 3, 1.4, 1.4]

# properties of PhC slab
a = 1
h = 1.375 * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(h, ep, fr, a)

k0a = 0.636 * 2 * np.pi
qa = 0 * 2 * np.pi
kya = 0.044 * 2 * np.pi
t1 = time.time()
cTIRfield = FieldsWithCTIR(phcs, k0a, qa, kya)
t2 = time.time()
print(t2 - t1)



