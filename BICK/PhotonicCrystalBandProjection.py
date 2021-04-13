import numpy as np
from MathTool import find_n_roots, find_real_roots, find_n_roots_for_small_and_big_q, find_real_roots_for_small_and_big_q
import matplotlib.pyplot as plt

def find_band_projection(phcs, n, Nq=100):

    ep = phcs.ep
    a = phcs.a
    fr = phcs.fr
    mu = phcs.mu
    nmax = np.sqrt(max(ep))

    k0_floor = []
    k0_ceiling = []
    dataq = []
    deltaq = deltak0 = 0.5 / Nq
    
    
    for ii in range(Nq - 1):
        
        q = (ii + 1) * deltaq
        qa = q * 2 * np.pi * a

        # dispersion relation of 1D PhC with kz=0
        def f_kza0(k0):
            k0a = k0 * a
            kxa = [(mu[i] * ep[i])**0.5 * k0a for i in range(2)]
            eta = (kxa[1] * mu[0]) / (kxa[0] * mu[1])
            output = np.cos(qa) - np.cos(kxa[0] * (1 - fr)) * np.cos(kxa[1] * fr) +\
                0.5 * (eta + 1 / eta) * np.sin(kxa[0] * (1 - fr)) * np.sin(kxa[1] * fr)
            return output.real
        # the k0 region of n_propagation progating modes and n_radiatoin
        if abs(ep[1] - ep[0]) * min([1 - fr, fr]) < 0.2:
            k0_start = find_n_roots_for_small_and_big_q(f_kza0, qa, n.real + 1, gox=1.0e-5, peak1=1.0e-5)
        else:
            k0_start = find_n_roots(f_kza0, n.real + 1, 0.12)
        if n.r % 2:
            k0_floor1 = np.max([k0_start[-2] / (2 * np.pi), q + (n.r - 1) / 2]) + deltak0
            k0_ceiling1 = np.min([k0_start[-1] / (2 * np.pi), (n.r + 1) / 2 - q])
        else:
            k0_floor1 = np.max([k0_start[-2] / (2 * np.pi), n.r / 2 - q]) + deltak0
            k0_ceiling1 = np.min([k0_start[-1] / (2 * np.pi), n.r / 2 + q])
        if k0_floor1 <= k0_ceiling1:
            k0_floor.append(k0_floor1)
            k0_ceiling.append(k0_ceiling1)
            dataq.append(q)
    
    plt.plot(dataq, k0_floor, 'b', ls=':')
    plt.plot(dataq, k0_ceiling, 'black', ls='--')
    plt.show()
    
    
    k0_max = max(k0_ceiling)
    k0_min = min(k0_floor)
    
    def f0(kza):
        kxa = [np.sqrt(mu[i] * ep[i] * (k0a) ** 2 - kza ** 2 + 0j) for i in range(2)]
        eta = (kxa[1] * mu[0]) / (kxa[0] * mu[1])
        output0 = - np.cos(kxa[0] * (1 - fr)) * np.cos(kxa[1] * fr) + \
            0.5 * (eta + 1 / eta) * np.sin(kxa[0] * (1 - fr)) * np.sin(kxa[1] * fr)
        return output0.real

    def f0i(kza):
        return f0(1j * kza)

    def f1(kza):
        output = 1 + f0(kza)
        return output.real

    def f2(kza):
        output = -1 + f0(kza)
        return output.real

    def fi1(kza):
        output = 1 + f0i(kza)
        return output.real

    def fi2(kza):
        output = -1 + f0i(kza)
        return output.real

    band_proj = {"k0a": np.arange(k0_min, k0_max, deltak0)}
    kpara_real_proj = []
    kpara_imag_proj = []
    for k0a in band_proj["k0a"]*2*np.pi*a:
        if abs(ep[1] - ep[0]) * min([1 - fr, fr]) < 0.2:
            kpara_real1 = np.array(find_real_roots_for_small_and_big_q(f1, 0))
            kpara_real2 = np.array(find_real_roots_for_small_and_big_q(f2, np.pi))
        else:
            kpara_real1 = np.array(find_real_roots(f1, nmax * k0a + 0.12))
            kpara_real2 = np.array(find_real_roots(f2, nmax * k0a + 0.12))
        
        temkpara_real_proj = np.concatenate((kpara_real1, kpara_real2))
        temkpara_real_proj.sort()
        kpara_real_proj.append(temkpara_real_proj.tolist())
        
        kpara_imag1 = np.array(find_n_roots_for_small_and_big_q(fi1, 0, n.imag + 1))
        kpara_imag2 = np.array(find_n_roots_for_small_and_big_q(fi2, np.pi, n.imag + 1))
        temkpara_real_proj = np.concatenate((kpara_imag1, kpara_imag2))
        temkpara_real_proj.sort()
        kpara_imag_proj.append(temkpara_real_proj.tolist())
        
    band_proj["real"] = np.array(kpara_real_proj)
    band_proj["imag"] = np.array(kpara_imag_proj)

    return k0_floor, k0_ceiling, band_proj
