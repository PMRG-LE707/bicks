import numpy as np
from bicks.mathtool import find_n_roots, find_real_roots
from bicks.mathtool import find_n_roots_for_small_and_big_q
from bicks.mathtool import find_real_roots_for_small_and_big_q

def find_band_projection(phcs, num, Nq=100, mode="E"):
    """
    find the area where we can use dichotomy to find roots(kz).
    
    Paramters
    ---------
    phcs: PhotonicCrystalSlab
        the Photonic Crystal Slab which is a kind of class.
    num: EssentialNumber
    mode: {"E", "H"}, optional
        the mode of the eigenstate   
    Nq: int, optional
        number which we divided half of Brillouin into
    
    Returns
    -------
    k0_floor: np.ndarray
        real k_parallels of eigenstates
    imag_k_parallel: np.ndarray
        imag k_parallels of eigenstates
    """
    
    a = phcs.a
    fr = phcs.fr
    if mode.lower() == "e":
        ep = phcs.ep
        mu = phcs.mu
    elif mode.lower() == "h":
        ep = -phcs.mu
        mu = -phcs.ep
    nmax = np.sqrt((ep*mu).max())

    k0_floor = []
    k0_ceiling = []
    dataq = []
    deltaq = deltak0 = 0.5 / Nq

    muep = mu * ep
    mu0, mu1 = mu
    ep0, ep1 = ep  
    for ii in range(Nq - 1):
        q = (ii + 1) * deltaq
        qa = q * 2 * np.pi * a
        # dispersion relation of 1D PhC with kz=0
        def f_kza0(k0):
            k0a = k0 * a
            kxa = np.sqrt(muep + 0j) * k0a 
            kxa0, kxa1 = kxa
            eta = (kxa1 * mu0) / (kxa0 * mu1)
            output = np.cos(qa) -\
            np.cos(kxa0 * (1 - fr)) * np.cos(kxa1 * fr) +\
            0.5 * (eta + 1 / eta) * np.sin(kxa0 * (1 - fr)) * np.sin(kxa1 * fr)
            return output.real
        # the k0 region of n_propagation progating modes and n_radiatoin
        if abs(ep1 - ep0) * min([1 - fr, fr]) < 0.2:
            k0_start = find_n_roots_for_small_and_big_q(
                    f_kza0, qa, num.real + 1, gox=1.0e-5, peak1=1.0e-5)
        else:
            k0_start = find_n_roots(f_kza0, num.real + 1, 0.12)
        if num.r % 2:
            k0_floor1 = np.max(
                    [k0_start[-2] / (2 * np.pi), q + (num.r - 1) / 2]) +\
                    deltak0
            k0_ceiling1 = np.min(
                    [k0_start[-1] / (2 * np.pi), (num.r + 1) / 2 - q])
        else:
            k0_floor1 = np.max(
                    [k0_start[-2] / (2 * np.pi), num.r / 2 - q]) + deltak0
            k0_ceiling1 = np.min(
                    [k0_start[-1] / (2 * np.pi), num.r / 2 + q])
        if k0_floor1 <= k0_ceiling1:
            k0_floor.append(k0_floor1)
            k0_ceiling.append(k0_ceiling1)
            dataq.append(q)
    
    dataq = np.array(dataq)
    k0_floor = np.array(k0_floor)
    k0_ceiling = np.array(k0_ceiling)
    k0_max = max(k0_ceiling)
    k0_min = min(k0_floor)
    

    def f0(kza):
        kxa = np.sqrt(muep * (k0a) ** 2 - kza ** 2 + 0j)
        kxa0, kxa1 = kxa
        eta = (kxa1 * mu0) / (kxa0 * mu1)
        output0 = - np.cos(kxa0 * (1 - fr)) * np.cos(kxa1 * fr) +\
            0.5 * (eta + 1 / eta) * np.sin(kxa0 * (1 - fr)) * np.sin(kxa1 * fr)
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
        if abs(ep1 - ep0) * min([1 - fr, fr]) < 0.2:
            kpara_real1 = np.array(
                    find_real_roots_for_small_and_big_q(f1, 0))
            kpara_real2 = np.array(
                    find_real_roots_for_small_and_big_q(f2, np.pi))
        else:
            kpara_real1 = np.array(find_real_roots(f1, nmax * k0a + 0.12))
            kpara_real2 = np.array(find_real_roots(f2, nmax * k0a + 0.12))
        
        temkpara_real_proj = np.concatenate((kpara_real1, kpara_real2))
        temkpara_real_proj.sort()
        kpara_real_proj.append(temkpara_real_proj.tolist())
        
        kpara_imag1 = np.array(
                find_n_roots_for_small_and_big_q(fi1, 0, num.imag + 1))
        kpara_imag2 = np.array(
                find_n_roots_for_small_and_big_q(fi2, np.pi, num.imag + 1))
        temkpara_real_proj = np.concatenate((kpara_imag1, kpara_imag2))
        temkpara_real_proj.sort()
        kpara_imag_proj.append(temkpara_real_proj.tolist())
        
    band_proj["real"] = kpara_real_proj
    band_proj["imag"] = kpara_imag_proj

    return k0_floor, k0_ceiling, dataq, band_proj

def mini_frequncy(phcs, num, qa, deltak0):
    """
    Find the floor of frequncy range for a specific q where the number of real
    k is a constant which is a paramter.
    
    Paramters:
    ----------
    phcs: PhotonicCrystalSlab
    num: EssentialNumber
    
    Returns
    -------
    k0_floor: np.ndarray
        the floor of range of k0 where the number of real k is a constant.
    """
    a = phcs.a
    fr = phcs.fr
    ep = phcs.ep
    mu = phcs.mu
    q = qa/(2*np.pi)    
    def find_k0_floor(ep, mu):
        muep = mu * ep
        mu0, mu1 = mu
        ep0, ep1 = ep
        def f_kza0(k0):
            k0a = k0 * a
            kxa = np.sqrt(muep + 0j) * k0a 
            kxa0, kxa1 = kxa
            eta = (kxa1 * mu0) / (kxa0 * mu1)
            output = np.cos(qa) -\
            np.cos(kxa0 * (1 - fr)) * np.cos(kxa1 * fr) +\
            0.5 * (eta + 1 / eta) * np.sin(kxa0 * (1 - fr)) * np.sin(kxa1 * fr)
            return output.real
        # the k0 region of n_propagation progating modes and n_radiatoin
        k0_start = find_n_roots_for_small_and_big_q(
                f_kza0, qa, num.real, gox=1.0e-10, peak1=1.0e-10)
        if num.r % 2:
            k0_floor = np.max(
                    [k0_start[-1] / (2 * np.pi), q + (num.r - 1) / 2]) +\
                    deltak0
        else:
            k0_floor = np.max(
                    [k0_start[-1] / (2 * np.pi), num.r / 2 - q]) + deltak0
        return k0_floor
    k0_floor1 = find_k0_floor(ep, mu)
    k0_floor2 = find_k0_floor(-mu, -ep)
    
    k0_floor = max([k0_floor1, k0_floor2])

    return k0_floor * (2 * np.pi)


