import numpy as np
from BoundryConditionWithcTIR import getcoefficents
from PhotonicCrystalSlab import PhotonicCrystalSlab
from eigenkpar import find_eigen_kpar
from Field import BulkEigenStates, FieldInPhcS, TotalFieldInPhcS, FiledsInAir


testi = -1
delnumber = 1

q = [0.1181, 0.1713, 0.072, 0.1311, 0.3156, 0.212]
k0 = [0.75398, 0.7632, 0.80759, 0.76285, 0.46092, 0.60472]
hset = [1.4, 2, 2.5, 3, 1.4, 1.4]

# properties of PhC slab
a = 1
h = 1.375 * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(h, ep, fr, a)


# properties of wavenumber


k0a = 0.636 * 2 * np.pi
qa = 0 * 2 * np.pi
kya = 0.044 * 2 * np.pi

nne = 3
npo = 3
nd = nne + npo + 1
nr = 1
npr = 2
# nEmode = nd + nr - npr + 1 for single p
nEmode = nd + 1
nHmode = nd + 1

if nEmode == 0:
    Ek_real_parallel, Ek_imag_parallel = [], []
else:
    Ek_real_parallel, Ek_imag_parallel = find_eigen_kpar(phcs, k0a, qa, 
                                                     nEmode, mode="E")
if nHmode == 0:
    Hk_real_parallel, Hk_imag_parallel = [], []
else:
    Hk_real_parallel, Hk_imag_parallel = find_eigen_kpar(phcs, k0a, qa, 
                                                     nHmode, mode="H")

Ek_real_parallel = np.delete(Ek_real_parallel,
                             1, axis=0)



E_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                      for kpar in Ek_real_parallel]
E_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                      for kpar in Ek_imag_parallel]

H_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                      for kpar in Hk_real_parallel]
H_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                      for kpar in Hk_imag_parallel]

real_eigenstates = E_real_eigenstates * 1
real_eigenstates.extend(H_real_eigenstates)

imag_eigenstates = E_imag_eigenstates * 1
imag_eigenstates.extend(H_imag_eigenstates)

imag_eigenstates = np.append(imag_eigenstates,
                             real_eigenstates[delnumber])

real_eigenstates = np.delete(real_eigenstates,
                             delnumber, axis=0)

real_fields_upward = [FieldInPhcS(eigenstate, kya=kya) 
                      for eigenstate in real_eigenstates]
imag_fields_upward = [FieldInPhcS(eigenstate, kya=kya) 
                      for eigenstate in imag_eigenstates]

real_fields_downward = [FieldInPhcS(eigenstate, kya=kya, kzdirection=-1)
                        for eigenstate in real_eigenstates]
imag_fields_downward = [FieldInPhcS(eigenstate, kya=kya, kzdirection=-1) 
                        for eigenstate in imag_eigenstates]

fields = [real_fields_upward, real_fields_downward, 
          imag_fields_upward, imag_fields_downward]
[coefs1, tx, ty], [coefs2, a2, a3] = getcoefficents(fields, nne, npo)
print(1 / coefs2[2])
print(coefs2[1] / coefs2[3])

