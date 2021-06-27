# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:29:16 2021

@author: CWXie
"""
from PhotonicCrystalBandProjection import mini_frequncy
from crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicky import FindBICs
from eigenkpar import find_eigen_kpar
import numpy as np
import matplotlib.pyplot as plt
from Field import FieldsWithCTIRMix

fr = 0.5
ep = [1, 4.9]
phcs = PhotonicCrystalSlab(ep, fr)
num = EssentialNumber(n_radiation=1)

deltak0 = 1.0e-3
k0f = mini_frequncy(phcs, num, 0, 0)
maxk0 = k0f + 0.5*2*np.pi
kpe = []
kph = []
numponit = 200
k0set = np.linspace(k0f, maxk0, num=numponit)
deltak0 = (maxk0 - k0f)/numponit
q = 0
for k0s in k0set:    
    singe_kpe = find_eigen_kpar(phcs, k0s, q, num.modes)
    singe_kph = find_eigen_kpar(phcs, k0s, q, num.modes, mode="H")
    kpe.append(singe_kpe)
    kph.append(singe_kph)
maxky = np.sqrt(maxk0**2 - q**2)
ky = 0

def radiationline(i, ky):
    if i%2:
        value_k0 = np.sqrt((q + (i - 1) / 2)**2 + ky**2)
    else:
        value_k0 = np.sqrt((i / 2 - q)**2 + ky**2)
    return value_k0

def num_positive(array, value):
    total = 0
    for item in array:
        if item>value:
            total = total + 1
    return total

xdata = []
ydata = []
xdata2 = []
ydata2 = []
xdata3 = []
ydata3 = []
alldata = []
while ky<maxky:
    kys = ky/(2*np.pi)
    k0f = radiationline(1, kys)*2*np.pi
    k0c = radiationline(2, kys)*2*np.pi
    for i in range(len(k0set)): 
        k0 = k0set[i]
        kpei = kpe[i][0]
        kphi = kph[i][0]
        if k0>k0f and k0<k0c:
            xdata.append(ky)
            ydata.append(k0)
            if num_positive(kpei, ky)==2 and num_positive(kphi, ky)==2:

                xdata2.append(ky)
                ydata2.append(k0)
                if kpei[0]<ky and kphi[0]<ky:
                    xdata3.append(ky)
                    ydata3.append(k0)
                    kppe = [kpei[1:], [kpei[0]]]
                    kpph = [kphi[1:], [kphi[0]]]
                    alldata.append([ky, k0, kppe, kpph])
                else:
                    alldata.append([ky, k0, kpe[i], kph[i]])
    ky = ky + deltak0
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.set_xlabel('$k_y$', font)
ax.set_ylabel('$k_0$', font)
ax.set_title("the range", font)
ax.scatter(xdata, ydata, s=0.1,
           label="1 ridiation channel")
ax.scatter(xdata2, ydata2, s=0.1,
           label="2 propagation modes(2 if $k_y=0$)")
ax.scatter(xdata3, ydata3, s=0.1, c="black",
           label="2 propagation modes(3 if $k_y=0$)")
plt.legend(markerscale=20)
plt.show()

for onedata in alldata:
    ky = onedata[0]
    k0 = onedata[1]
    kparae = onedata[2]
    kparah = onedata[3]
    f1 = FieldsWithCTIRMix(phcs, num, k0, 0, ky, kparae, kparah)
    numb = 0
    for coef in [f1.even_coefs_inside[0],f1.even_coefs_inside[2]]:
        if abs(coef.real)>0.999:
            numb = numb + 1
    if numb == 2:
        print("=====")
        print(f1.even_coefs_inside)
        print("h: ",phcs.h)
        print("k0: ", k0/(2*np.pi))
        print("ky: ", ky/(2*np.pi))
    



