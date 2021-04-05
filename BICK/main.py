import numpy as np
from BoundryConditionWithcTIR import getcoefficents
from PhotonicCrystalSlab import PhotonicCrystalSlab
from eigenkpar import find_eigen_kpar
from Field import BulkEigenStates, FieldInPhcS, TotalFieldInPhcS, FiledsInAir


# properties of PhC slab
a = 1
h = 1.4 * a
fr = 0.5
ep = [1.0, 4.9]
phcs = PhotonicCrystalSlab(h, ep, fr, a)

# properties of wavenumber
k0a = 0.6047 * 2 * np.pi
qa = 0.0
kya = 0.212 * 2 * np.pi

nne = 2
npo = 2
nd = nne + npo + 1
nEmode = nd
nHmode = nd - 1

Ek_real_parallel, Ek_imag_parallel = find_eigen_kpar(phcs, k0a, qa, 
                                                     nEmode, mode="E")
Hk_real_parallel, Hk_imag_parallel = find_eigen_kpar(phcs, k0a, qa, 
                                                     nHmode, mode="H")

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
coefs, tx, ty = getcoefficents(fields, nne, npo)

totalfield = []
for field in fields:
    totalfield.extend(field)
totalfieldinPhC = TotalFieldInPhcS(totalfield, coefs)

totalfieldinair = FiledsInAir(tx, ty, nne, npo, qa, k0a, kya)
