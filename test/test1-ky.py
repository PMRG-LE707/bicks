# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:29:16 2021

@author: CWXie
"""
from bicky.photoniccrystalbandprojection import mini_frequncy
from bicky.crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicky.eigenkpar import find_eigen_kpar
import numpy as np
import matplotlib.pyplot as plt
from bicky.field import FieldsWithCTIRMix

n_imag_p = 2
fr = 0.5
ep = [1, 4.9]
phcs = PhotonicCrystalSlab(ep, fr, thickness=1.492)
num = EssentialNumber(n_radiation=1, nimag_plus=n_imag_p)

deltak0 = 1.0e-3
k0f = mini_frequncy(phcs, num, 0, 0)
maxk0 = k0f + 0.5*2*np.pi
kpe = []
kph = []
numponit = 200
k0set = np.linspace(k0f, maxk0, num=numponit)
deltak0 = (maxk0 - k0f)/numponit
q = 0 * 2 * np.pi
for k0s in k0set:    
    singe_kpe = find_eigen_kpar(phcs, k0s, q, num.modes)
    singe_kph = find_eigen_kpar(phcs, k0s, q, num.modes, mode="H")
    kpe.append(singe_kpe)
    kph.append(singe_kph)
maxky = np.sqrt(maxk0**2 - q**2)
ky = 0

def radiationline(i, ky):
    if i%2:
        value_k0 = np.sqrt((q/(2*np.pi) + (i - 1) / 2)**2 + ky**2)
    else:
        value_k0 = np.sqrt((i / 2 - q/(2*np.pi))**2 + ky**2)
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
            xdata.append(ky/(2*np.pi))
            ydata.append(k0/(2*np.pi))
            if num_positive(kpei, ky)==2 and num_positive(kphi, ky)==2:

                xdata2.append(ky/(2*np.pi))
                ydata2.append(k0/(2*np.pi))
                if kpei[0]<ky and kphi[0]<ky:
                    xdata3.append(ky/(2*np.pi))
                    ydata3.append(k0/(2*np.pi))
                    imkpe = kpe[i][1]*1
                    imkph = kph[i][1]*1
                    imkpe.append(kpei[0])
                    imkph.append(kphi[0])
                    kppe = [kpei[1:], imkpe]
                    kpph = [kphi[1:], imkph]
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

xdata4 = []
ydata4 = []
for i in range(len(alldata)):
    onedata = alldata[i]
    ky = onedata[0]
    k0 = onedata[1]
    kparae = onedata[2]
    kparah = onedata[3]
    f1 = FieldsWithCTIRMix(phcs, num, k0, 0, ky, kparae, kparah)
    numb = 0
    if f1.even_coefs_inside[1].real<-0.999 and\
    f1.even_coefs_inside[2].real<-0.999:
        numb = numb + 2
    if numb == 2:
        
        print("=====")
        print("nimag: ", n_imag_p)
        print(f1.even_coefs_inside)
        print("h: ",phcs.h)
        print("k0: ", k0/(2*np.pi))
        print("ky: ", ky/(2*np.pi))
        
ax.scatter(xdata4, ydata4, s=1,
           label="parallal")
plt.legend(markerscale=1)
plt.show()