# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy
from bicks.boundryconditionwithcTIR import getcoesingle,  getcoemix

class BulkEigenStates:
    """
    The eigenstates i.e. periodic parts of fields in PhC
    """
    def __init__(self, phcs, k0a, kpar, qa, mode="E", normalization=1):
        """Initialize the eigenstates in PhC
        
        Paramters
        ---------
        phcs: PhotonicCrystalSlab
            the Photonic Crystal Slab which is a kind of class.
        k0a: float
            the wave number
        kpar: complex
            the k_parallel of Bloch mode
        qa: float
            the Bloch wave number q * a
        mode: {"E", "H"}, optional
            the mode of the eigenstate
        normalization: complex, optional
            normalization factor
        
        """
        if mode == "H":
            mu = -np.array(phcs.ep)
            ep = -np.array(phcs.mu)
        else:
            mu = phcs.mu
            ep = phcs.ep
        newphcs = copy.deepcopy(phcs)
        newphcs.mu = mu
        newphcs.ep = ep
        fr = phcs.fr
        # ky * a in the homogeneous medium(2 layers)
        kxa_ho_med = np.array([np.sqrt(mu[j] * ep[j] * (k0a) ** 2
                                       - kpar ** 2 + 0j)
                               for j in range(2)])
        
        eta1 = (kxa_ho_med[1] / kxa_ho_med[0]) * (mu[0] / mu[1])
        eta2 = 1 / eta1
        eigenvalue = np.exp(1j * qa)
        
        pd1 = np.array([[np.exp(-1j * kxa_ho_med[0] * (1 - fr)), 0], 
                        [0, np.exp(1j * kxa_ho_med[0] * (1 - fr))]])
        d12 = np.array([[(1 + eta1) * 0.5, (1 - eta1) * 0.5], 
                        [(1 - eta1) * 0.5, (1 + eta1) * 0.5]])
        pd2 = np.array([[np.exp(-1j * kxa_ho_med[1] * fr), 0], 
                        [0, np.exp(1j * kxa_ho_med[1] * fr)]])
        d21 = np.array([[(1 + eta2) * 0.5, (1 - eta2) * 0.5], 
                        [(1 - eta2) * 0.5, (1 + eta2) * 0.5]])
        pdd = np.dot(pd1, d12)
        pddpd2 = np.dot(pdd, pd2)
        m = np.dot(pddpd2, d21)  
        inverspdd = np.array([[pdd[1, 1], -pdd[0, 1]],
                              [-pdd[1, 0], pdd[0, 0]]])\
            /(-pdd[0, 1] * pdd[1, 0] + pdd[0, 0] * pdd[1, 1])
        a0 = 1
        b0 = (1 - eigenvalue * m[0, 0]) / (eigenvalue * m[0, 1])
        c0 = a0 * inverspdd[0, 0] + b0 * inverspdd[0, 1]
        d0 = a0 * inverspdd[1, 0] + b0 * inverspdd[1, 1]
        
        self.k0a = k0a
        self.kpar = kpar
        self.kxa = kxa_ho_med
        self.qa = qa
        self.mode = mode
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
        self.phcs = newphcs
        self.normalization = normalization
    
    def u(self, xra):
        if xra < (1 - self.phcs.fr):
            output = self.a0 * np.exp(1j * self.kxa[0] * xra)\
                + self.b0 * np.exp(-1j * self.kxa[0] * xra)
        else:
            nxra = xra - (1 - self.phcs.fr)
            output = self.c0 * np.exp(1j * self.kxa[1] * nxra)\
                + self.d0 * np.exp(-1j * self.kxa[1] * nxra)
        return output * np.exp(-1j * self.qa * xra) / self.normalization
    
    def w(self, xra):
        if xra < (1 - self.phcs.fr):
            output = self.kxa[0] / self.phcs.mu[0]\
                * (self.a0 * np.exp(1j * self.kxa[0] * xra) - 
                   self.b0 * np.exp(-1j * self.kxa[0] * xra))
        else:
            nxra = xra - (1 - self.phcs.fr)
            output = self.kxa[1] / self.phcs.mu[1]\
                * (self.c0 * np.exp(1j * self.kxa[1] * nxra) - 
                   self.d0 * np.exp(-1j * self.kxa[1] * nxra))
        return output * np.exp(-1j * self.qa * xra) / self.normalization
    
    def v(self, xra):
        if xra < (1 - self.phcs.fr):
            output = 1 / self.phcs.mu[0] * self.u(xra)
        else:
            output = 1 / self.phcs.mu[1] * self.u(xra)
        return output * np.exp(-1j * self.qa * xra) / self.normalization
    
    def Fourier_coefficients(self, i):
        kxa = self.kxa
        qa = self.qa
        fr = self.phcs.fr
        mu = self.phcs.mu

        ka = -i * 2 * np.pi + kxa[0] - qa
        kb = -i * 2 * np.pi - kxa[0] - qa
        kc = -i * 2 * np.pi + kxa[1] - qa
        kd = -i * 2 * np.pi - kxa[1] - qa
        Aa = -1j * self.a0 / ka * (np.exp(1j * (1 - fr) * ka) - 1)
        Ab = -1j * self.b0 / kb * (np.exp(1j * (1 - fr) * kb) - 1)
        Ac = -1j * self.c0 / kc * (np.exp(1j * kc) -
                                   np.exp(1j * (1 - fr) * kc))\
            * np.exp(-1j * kxa[1] * (1 - fr))
        Ad = -1j * self.d0 / kd * (np.exp(1j * kd) - 
                                   np.exp(1j * (1 - fr) * kd))\
            * np.exp(1j * kxa[1] * (1 - fr))
        A = Aa + Ab + Ac + Ad
        B = kxa[0] / mu[0] * (Aa - Ab) + kxa[1]  / mu[1] * (Ac - Ad)
        C = 1 / mu[0] * (Aa + Ab) + 1 / mu[1] * (Ac + Ad)

        A = A / self.normalization
        B = B / self.normalization
        C = C / self.normalization
        return A, B, C

class FieldInPhcS:
    """
    The field generated by a BulkEigenStates in PhC.
    """
    def __init__(self, eigenstate, kya=0):    
        """Initialize the field-in-PhC
        
        Paramters
        ---------
        eigenstate: BulkEigenStates
            a bulk state to generate the field
        kya: float, Optional
            ky * a
        """
        self.kya = kya
        self.kza = np.array([np.sqrt(eigenstate.kpar ** 2 
                                     - kya ** 2 + 0j),
                             -np.sqrt(eigenstate.kpar ** 2 
                                     - kya ** 2 + 0j)])
        self.es = eigenstate
        self.k0a = eigenstate.k0a
        self.qa = eigenstate.qa
        self.mode = eigenstate.mode
        
    def Ex(self, x, z, kzdirection=0):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza[0]
        if self.mode == "E":
            output = 0
        else:
            output = -es.v(x) * np.exp(1j * kza * z)\
                * (kya ** 2 / kza + kza) / k0a
        return output * np.exp(1j * qa * x)
    
    def Ey(self, x, z, kzdirection=0):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza[0]
        if es.mode == "E":
            output = es.u(x) * np.exp(1j * kza * z)
        else:
            output = es.w(x) * np.exp(1j * kza * z) * kya / (k0a * kza)
        return output * np.exp(1j * qa * x)
        
    def Hx(self, x, z, kzdirection=0):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza[0]
        if es.mode == "H":
            output = 0
        else:
            output = -es.v(x) * np.exp(1j * kza * z)\
                * (kya ** 2 / kza + kza) / k0a
        return output * np.exp(1j * qa * x)
    
       
    def Hy(self, x, z, kzdirection=0):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza[0]
        if es.mode == "H":
            output = es.u(x) * np.exp(1j * kza * z)
        else:
            output = es.w(x) * np.exp(1j * kza * z) * kya / (k0a * kza)
        return output * np.exp(1j * qa * x)

  
    def fieldfc(self, i, field_compoments):
        """
        without the basis
        """
        es = self.es
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        z = es.phcs.h / 2
        A, B, C = es.Fourier_coefficients(i)
        
        expz = np.exp(1j * kza * z)
        field_Fourier = {}
        for field_direction in field_compoments:
            if field_direction == "Ex":
                if self.mode == "E":
                    output = 0 * kza
                else:
                    output = -C * expz * (kya ** 2 / kza + kza) / k0a
 
            elif field_direction == "Ey":
                if es.mode == "E":
                    output = A * expz
                else:
                    output = B * expz * kya / (k0a * kza)
 
            elif field_direction == "Hx":
                if es.mode == "H":
                    output = 0 * kza
                else:
                    output = -C * expz * (kya ** 2 / kza + kza) / k0a
 
            else:
                if es.mode == "H":
                    output = A * expz
                else:
                    output = B * expz * kya / (k0a * kza)
            
            field_Fourier[field_direction] = output
        self.field_Fourier = field_Fourier
        return field_Fourier
    
    def show(self, fieldcomponent, oprator, Nx=20):
        """
        unit in period a
        """
        if self.mode == "E":
            ep = self.es.phcs.ep
        else:
            ep = -self.es.phcs.mu
        a = self.es.phcs.a
        hight = self.es.phcs.h / a
        fr = self.es.phcs.fr
        width1 = (1 - fr)
        width2 = fr
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('x (a)', font)
        ax1.set_ylabel('z (a)', font)
        ax1.set_title(oprator + " $" + fieldcomponent[0] + 
                      "_{" + fieldcomponent[1]+ "}$ in one period" + 
                      "(bulk state)", font)
        ax1.set_xlim([-0.5, 1.5])
        ax1.set_ylim([-hight * 0.6, hight])
        
        Nz = int(hight * Nx)
        xx = np.arange(0, 2 * Nx + 1) * (1 / (2 * Nx))
        zz = np.arange(-Nz, Nz + 1) * (1 / (2 * Nx))
        oprator = eval("np." + oprator)
        fieldcomponent = eval("self." + fieldcomponent)
        value = oprator(np.array([[fieldcomponent(x, z) 
                                   for x in xx]
                                  for z in zz]))
        
        profile = ax1.imshow(value, cmap='RdBu', 
                         interpolation='none', 
                         extent=[0, 1, hight * 0.5, -hight * 0.5])
        
        fig.colorbar(profile, extend='both')
        ax1.plot([1 - fr, 1 - fr], [-hight / 2, hight / 2],
                 color="Black", linestyle='dashed') 
        
        # plot the arrow and delectric constants
        txtname = ['$\epsilon_{1} = ' + str(ep[0]) + '$',
                   '$\epsilon_{2} = ' + str(ep[1]) + '$']
        ax1.annotate(txtname[0],
            xy=(width1/2, hight / 2 * 0.9), xycoords='data',
            xytext=(0.1, 0.82), textcoords='axes fraction',
            bbox=dict(boxstyle="round4", fc="maroon", alpha=0.3),
            arrowprops=dict(arrowstyle="fancy",
                                  connectionstyle="arc3,rad=0.2",
                                  fc="maroon", alpha=0.3))
        ax1.annotate(txtname[1],
            xy=(width1 + width2 / 2, hight / 2 * 0.9), xycoords='data',
            xytext=(0.7, 0.82), textcoords='axes fraction',
            bbox=dict(boxstyle="round4",  fc="blue", alpha=0.3),
            arrowprops=dict(arrowstyle="fancy",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="blue", alpha=0.3))
        kz = self.kza / (2 * np.pi * a)
        kz = round(kz, 3)
        if np.imag(kz) == 0:
            kz = kz.real
            kz = str(kz)
        else:
            kz = kz.imag
            kz = str(kz)+"i"
        ax1.text(0.5, 0.9 * hight, 
                 str(self.mode) + " mode: $kz = " + kz + " (2\pi/a)$", 
                 size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
        plt.show()
        return
     
class FieldsWithCTIRMix:
    
    def __init__(self, phcs, num,
                 k0a, qa, kya,
                 kpe, kph):
        """
        there is one more propagating modes than radiation channels.
        """

        self.phcs = phcs
        self.k0a = k0a
        self.qa = qa
        self.kya = kya
        
        Ek_real_parallel = kpe[0]
        Ek_imag_parallel = kpe[1]
        Hk_real_parallel = kph[0]
        Hk_imag_parallel = kph[1]
        E_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                              for kpar in Ek_real_parallel]
        E_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                              for kpar in Ek_imag_parallel]
        
        H_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                              for kpar in Hk_real_parallel]
        H_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                              for kpar in Hk_imag_parallel]
        sw_num = 0
        sw_eigenstates = E_real_eigenstates[sw_num]
        del E_real_eigenstates[sw_num]
        E_imag_eigenstates.append(sw_eigenstates)
        real_eigenstates = E_real_eigenstates * 1
        real_eigenstates.extend(H_real_eigenstates)
        
        imag_eigenstates = E_imag_eigenstates * 1
        imag_eigenstates.extend(H_imag_eigenstates)
        
        real_fields = [FieldInPhcS(eigenstate, kya=kya) 
                              for eigenstate in real_eigenstates]
        imag_fields = [FieldInPhcS(eigenstate, kya=kya) 
                              for eigenstate in imag_eigenstates]
        
        even_coefs, odd_coefs, real_kzas = \
            getcoemix(real_fields, imag_fields, num)
        self.even_coefs_inside = np.array(even_coefs)
        self.odd_coefs_inside = np.array(odd_coefs)
        self.realkzs = real_kzas
        

class FieldsWithCTIRInArea:
    """
    The class can get the coefficinents of different fields
    when the cTIR happends on the upper boundry in specific
    q-k0 area dicided by EssentialNumber num. 
    """
    def __init__(self, phcs, num,
                 k0a, qa, kya,
                 real_parallel,
                 imag_parallel,
                 mode="E"):
        """
        Initialize the field-in-PhC.
        There is one more propagating mode than radiation channels.
        
        Paramters
        ---------
        phcs: PhotonicCrystalSlab
            the Photonic Crystal Slab which is a kind of class.
        k0a: float
            the frequency divided by (2pi*c)
        kya: float
            ky * a
        qa: float
            the Bloch wave q * a
        num: EssentialNumber
            essentialnumber
        real_parallel: complex
            the real k_parallel of Bloch modes
        imag_parallel: complex
            the imag k_parallel of Bloch modes
        mode: {"E", "H"}, optional
            the mode of the eigenstate
        """
        self.phcs = phcs
        self.k0a = k0a
        self.qa = qa
        self.kya = kya
        
        
        if mode.lower() == "e":
            nEmode = num.modes
        elif mode.lower() == "h":
            nEmode = 0
        else:
            raise ValueError("mode should be 'E' or 'H'")
        
        if nEmode == 0:
            Ek_real_parallel, Ek_imag_parallel = [], []
            Hk_real_parallel, Hk_imag_parallel = \
                real_parallel, imag_parallel
        else:
            Hk_real_parallel, Hk_imag_parallel = [], []
            Ek_real_parallel, Ek_imag_parallel = \
                real_parallel, imag_parallel
                
        E_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                              for kpar in Ek_real_parallel]
        E_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="E") 
                              for kpar in Ek_imag_parallel]
        
        H_real_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                              for kpar in Hk_real_parallel]
        H_imag_eigenstates = [BulkEigenStates(phcs, k0a, kpar, qa, mode="H") 
                              for kpar in Hk_imag_parallel]
        
        real_eigenstates = E_real_eigenstates
        real_eigenstates.extend(H_real_eigenstates)
        
        imag_eigenstates = E_imag_eigenstates
        imag_eigenstates.extend(H_imag_eigenstates)
        
        real_fields = [FieldInPhcS(eigenstate, kya=kya) 
                              for eigenstate in real_eigenstates]
        imag_fields = [FieldInPhcS(eigenstate, kya=kya) 
                              for eigenstate in imag_eigenstates]
        
        even_coefs, odd_coefs, real_kzas = \
            getcoesingle(real_fields, imag_fields, num)
        
        self.even_coefs_inside = np.array(even_coefs)
        self.odd_coefs_inside = np.array(odd_coefs)
        self.realkzs = real_kzas