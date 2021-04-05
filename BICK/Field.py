# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy


class BulkEigenStates:
    """
    the eigen states i.e. periodic parts of fields
    """
    def __init__(self, phcs, k0a, kpar, qa, mode="E", normalization=1):
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
        kxa_ho_med = np.array([np.sqrt(mu[j] * ep[j] * (k0a) ** 2 - kpar ** 2 + 0j) for j in range(2)])
        
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
        inverspdd = np.array([[pdd[1, 1], -pdd[0, 1]], [-pdd[1, 0], pdd[0, 0]]])/(-pdd[0, 1] * pdd[1, 0] + pdd[0, 0] * pdd[1, 1])
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
            output = self.a0 * np.exp(1j * self.kxa[0] * xra) + self.b0 * np.exp(-1j * self.kxa[0] * xra)
        else:
            nxra = xra - (1 - self.phcs.fr)
            output = self.c0 * np.exp(1j * self.kxa[1] * nxra) + self.d0 * np.exp(-1j * self.kxa[1] * nxra)
        return output * np.exp(-1j * self.qa * xra) / self.normalization
    
    def w(self, xra):
        if xra < (1 - self.phcs.fr):
            output = self.kxa[0] / self.phcs.mu[0] * (self.a0 * np.exp(1j * self.kxa[0] * xra) - self.b0 * np.exp(-1j * self.kxa[0] * xra))
        else:
            nxra = xra - (1 - self.phcs.fr)
            output = self.kxa[1] / self.phcs.mu[1] * (self.c0 * np.exp(1j * self.kxa[1] * nxra) - self.d0 * np.exp(-1j * self.kxa[1] * nxra))
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
        Ac = -1j * self.c0 / kc * (np.exp(1j * kc) - np.exp(1j * (1 - fr) * kc)) * np.exp(-1j * kxa[1] * (1 - fr))
        Ad = -1j * self.d0 / kd * (np.exp(1j * kd) - np.exp(1j * (1 - fr) * kd)) * np.exp(1j * kxa[1] * (1 - fr))
        A = Aa + Ab + Ac + Ad
        B = kxa[0] / mu[0] * (Aa - Ab) + kxa[1]  / mu[1] * (Ac - Ad)
        C = 1 / mu[0] * (Aa + Ab) + 1 / mu[1] * (Ac + Ad)

        A = A / self.normalization
        B = B / self.normalization
        C = C / self.normalization
        return A, B, C
    

class TotalFieldInPhcS:
    def __init__(self, fields, coefs):
        self.fields = fields
        self.coefs = coefs
        
    def Ex(self, x, z):
        
        fields = self.fields
        coefs = self.coefs
        sumEx = 0
        for i in range(len(fields)):
            field = coefs[i] * fields[i].Ex(x, z)
            sumEx = sumEx + field
        return sumEx
    
    def Ey(self, x, z):
        
        fields = self.fields
        coefs = self.coefs
        sumEy = 0
        for i in range(len(fields)):
            field = coefs[i] * fields[i].Ey(x, z)
            sumEy = sumEy + field
        return sumEy
        
    def Hx(self, x, z):
        
        fields = self.fields
        coefs = self.coefs
        sumHx = 0
        for i in range(len(fields)):
            field = coefs[i] * fields[i].Hx(x, z)
            sumHx = sumHx + field
        return sumHx
    
    def Hy(self, x, z):
        
        fields = self.fields
        coefs = self.coefs
        sumHy = 0
        for i in range(len(fields)):
            field = coefs[i] * fields[i].Hy(x, z)
            sumHy = sumHy + field
        return sumHy
  
    def zenergy_flow(self, z, n_x=1000, max_x=1):
        """
        energy flow in the direction z on the z=z(a whole period)
        """
        delta_x = max_x/n_x
        # set up Energy Flow
        EF = 0
    
        for i in range(n_x):
            # energy flow density
            x = i * delta_x
            Sz = np.real(self.Ex(x, z) * np.conj(self.Hy(x, z)) - self.Ey(x, z) * np.conj(self.Hx(x, z)))
            EF = EF + Sz * delta_x
            return EF

class FieldInPhcS:
    def __init__(self, eigenstate, kya=0, kzdirection=1):    
        self.kya = kya
        self.kza = kzdirection * np.sqrt(eigenstate.kpar ** 2 - kya ** 2 + 0j)
        self.es = eigenstate
        self.k0a = eigenstate.k0a
        self.qa = eigenstate.qa
        self.mode = eigenstate.mode
        
    def Ex(self, x, z):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        if self.mode == "E":
            output = 0
        else:
            output = -es.v(x) * np.exp(1j * kza * z) * (kya ** 2 / kza + kza) / k0a
        return output * np.exp(1j * qa * x)
    
    def Ey(self, x, z):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        if es.mode == "E":
            output = es.u(x) * np.exp(1j * kza * z)
        else:
            output = es.w(x) * np.exp(1j * kza * z) * kya / (k0a * kza)
        return output * np.exp(1j * qa * x)
        
    def Hx(self, x, z):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        if es.mode == "H":
            output = 0
        else:
            output = -es.v(x) * np.exp(1j * kza * z) * (kya ** 2 / kza + kza) / k0a
        return output * np.exp(1j * qa * x)
    
       
    def Hy(self, x, z):
        es = self.es
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        if es.mode == "H":
            output = es.u(x) * np.exp(1j * kza * z)
        else:
            output = es.w(x) * np.exp(1j * kza * z) * kya / (k0a * kza)
        return output * np.exp(1j * qa * x)

  
    def fieldfc(self, i, field_direction):
        """
        without the basis
        """
        es = self.es
        k0a = self.k0a
        kya = self.kya
        kza = self.kza
        z = es.phcs.h / 2
        A, B, C = es.Fourier_coefficients(i)
        
        if field_direction == "Ex":
            if self.mode == "E":
                output = 0
            else:
                output = -C * np.exp(1j * kza * z) * (kya ** 2 / kza + kza) / k0a
            return output
        
        elif field_direction == "Ey":
            if es.mode == "E":
                output = A * np.exp(1j * kza * z)
            else:
                output = B * np.exp(1j * kza * z) * kya / (k0a * kza)
            return output
        
        elif field_direction == "Hx":
            if es.mode == "H":
                output = 0
            else:
                output = -C * np.exp(1j * kza * z) * (kya ** 2 / kza + kza) / k0a
            return output
        
        else:
            if es.mode == "H":
                output = A * np.exp(1j * kza * z)
            else:
                output = B * np.exp(1j * kza * z) * kya / (k0a * kza)
            return output
    
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
        print(xx)
        oprator = eval("np." + oprator)
        fieldcomponent = eval("self." + fieldcomponent)
        value = oprator(np.array([[fieldcomponent(x, z) for x in xx] for z in zz]))
        print(oprator(fieldcomponent(0.6, -0.4)))
        
        profile = ax1.imshow(value, cmap='RdBu', 
                         interpolation='none', 
                         extent=[0, 1, hight * 0.5, -hight * 0.5])
        
        fig.colorbar(profile, extend='both')
        ax1.plot([1 - fr, 1 - fr], [-hight / 2, hight / 2], color="Black", linestyle='dashed') 
        
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
    
class FiledsInAir:
    def __init__(self, tx, ty, nne, npo, qa, k0a, kya):
        self.tx = tx
        self.ty = ty
        self.nne = nne
        self.npo = npo
        self.qa = qa
        self.k0a = k0a
        self.kya = kya
        
    def Ex(self, x, z):
        tx = self.tx
        nne = self.nne
        npo = self.npo
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        sumEx = 0
        for i in range(-nne, npo + 1):
            kxai = i * 2 * np.pi + qa
            kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
            if i != 0:
                expz = np.exp(1j * kzaouti * z)
                expx = np.exp(1j * kxai * x)
                sumEx = sumEx + tx[i] * expz * expx
        return sumEx
    
    def Ey(self, x, z):
        ty = self.ty
        nne = self.nne
        npo = self.npo
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        sumEy = 0
        for i in range(-nne, npo + 1):
            kxai = i * 2 * np.pi + qa
            kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
            if i != 0:
                expz = np.exp(1j * kzaouti * z)
                expx = np.exp(1j * kxai * x)
                sumEy = sumEy + ty[i] * expz * expx
        return sumEy
    
    def Hx(self, x, z):
        tx = self.tx
        ty = self.ty
        nne = self.nne
        npo = self.npo
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        sumHx = 0
        for i in range(-nne, npo + 1):
            kxai = i * 2 * np.pi + qa
            kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
            if i != 0:
                expz = np.exp(1j * kzaouti * z)
                expx = np.exp(1j * kxai * x)
                sumHx = sumHx - (tx[i] * kxai * kya / kzaouti + 
                                 ty[i] * (kya ** 2 / kzaouti + kzaouti)) / k0a * expz * expx
        return sumHx
    
    def Hy(self, x, z):
        tx = self.tx
        ty = self.ty
        nne = self.nne
        npo = self.npo
        qa = self.qa
        k0a = self.k0a
        kya = self.kya
        sumHy = 0
        for i in range(-nne, npo + 1):
            kxai = i * 2 * np.pi + qa
            kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
            if i != 0:
                expz = np.exp(1j * kzaouti * z)
                expx = np.exp(1j * kxai * x)
                sumHy = sumHy - (ty[i] * kxai * kya / kzaouti + 
                                 tx[i] * (kxai ** 2 / kzaouti + kzaouti)) / k0a * expz * expx
        return sumHy
        
        
   