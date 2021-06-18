# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:29:16 2021

@author: CWXie
"""
from PhotonicCrystalSlab import PhotonicCrystalSlab, EssentialNumber
from bicky import FindBICs
import time

t1 = time.time()

a = 1
fr = 0.5
ep = [1, 4.9]
phcs = PhotonicCrystalSlab(ep, fr, a)
num = EssentialNumber(n_radiation=1)

fb = FindBICs(phcs, num)
fb.getcoeffs()

t2 = time.time()
print(t2 - t1)
hstart = 1.2
hend = 2.5
Nh = 100
fb.run(hstart, hend, Nh)