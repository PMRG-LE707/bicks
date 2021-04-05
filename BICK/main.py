import numpy as np
from BoundryConditionWithcTIR import getcoefficents
from PhotonicCrystalSlab import PhotonicCrystalSlab
from eigenkpar import find_eigen_kpar
from Field import BulkEigenStates, FieldInPhcS


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
nEmode = nd + 1
nHmode = nd

Ek_real_parallel, Ek_imag_parallel = find_eigen_kpar(phcs, k0a, qa, nEmode, mode="E")
Hk_real_parallel, Hk_imag_parallel = find_eigen_kpar(phcs, k0a, qa, nHmode, mode="H")

E_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") for kpar in Ek_real_parallel]
E_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") for kpar in Ek_imag_parallel]

H_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") for kpar in Hk_real_parallel]
H_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") for kpar in Hk_imag_parallel]

eigenstates = E_real_eigenstates * 1
eigenstates.extend(H_real_eigenstates)
eigenstates.extend(E_imag_eigenstates)
eigenstates.extend(H_imag_eigenstates)

fields_upward = [FieldInPhcS(eigenstate, kya=kya) for eigenstate in eigenstates]
fields_downward = [FieldInPhcS(eigenstate, kya=kya, kzdirection=-1) for eigenstate in eigenstates]

mm = getcoefficents(fields_upward, fields_downward, nne, npo)
mm = np.around(mm, 3)