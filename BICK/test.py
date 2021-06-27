# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:29:16 2021

@author: CWXie
"""
from PhotonicCrystalBandProjection import mini_frequncy
from crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicky import FindBICs
from eigenkpar import find_eigen_kpar

fr = 0.5
ep = [1, 4.9]
phcs = PhotonicCrystalSlab(ep, fr)
num = EssentialNumber(n_radiation=1)

k0f = mini_frequncy(phcs, num, 0, 0.01)