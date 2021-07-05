# -*- coding: utf-8 -*-
from bicky.photoniccrystalbandprojection import mini_frequncy
from bicky.crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicky.eigenkpar import find_eigen_kpar
import numpy as np
import matplotlib.pyplot as plt
from bicky.field import FieldsWithCTIRMix

n_imag_p = 2
fr = 0.5
ep = [1, 4.9]
phcs = PhotonicCrystalSlab(ep, fr, thickness=1.5)
num = EssentialNumber(n_radiation=1, nimag_plus=n_imag_p)
deltak0 = 1.0e-3
k0f = mini_frequncy(phcs, num, 0, 0)
maxk0 = k0f + 0.5*2*np.pi
ky = 0.26
k0 = 0.7*2*np.pi
singe_kpe = find_eigen_kpar(phcs, k0, 0, num.modes)
singe_kph = find_eigen_kpar(phcs, k0, 0, num.modes, mode="H")
f1 = FieldsWithCTIRMix(phcs, num, k0, 0, ky, singe_kpe, singe_kph)
print(f1.even_coefs_inside)