import numpy as np
from bicks.mathtool import find_real_roots_for_small_and_big_q, \
find_n_roots_for_small_and_big_q, find_real_roots, dichotomy

def find_eigen_kpar(phcs, k0a, qa, nmode, mode="E"):
    """The eigenstates of the 1D photonic crystal.
    
    Paramters
    ---------
    phcs: PhotonicCrystalSlab
        the Photonic Crystal Slab which is a kind of class.
    qa: float
        the Bloch wave number
    k0a: float
        the frequency divided by (2pi*c)
    nmode: int
        the number of considered Bloch modes
    mode: {"E", "H"}, optional
        the mode of the eigenstate
    
    Returns
    -------
    np.ndarray
        real k_parallels(eigenvalue) of eigenstates
    np.ndarray
        imaginary k_parallels(eigenvalue) of eigenstates
    """

    fr = phcs.fr
    if mode.lower() == "e":
        ep = phcs.ep
        mu = phcs.mu
    elif mode.lower() == "h":
        ep = -phcs.mu
        mu = -phcs.ep
    nmax = np.sqrt((ep*mu).max())
    
    def f(k_parallel):
        kxa = [np.sqrt(mu[i] * ep[i] * (k0a) ** 2 - k_parallel ** 2 + 0j)\
               for i in range(2)]
        eta = (kxa[1] * mu[0]) / (kxa[0] * mu[1])
        output = np.cos(qa) - np.cos(kxa[0] * (1 - fr)) * np.cos(kxa[1] * fr)\
        + 0.5 * (eta + 1 / eta) * np.sin(
            kxa[0] * (1 - fr)) * np.sin(kxa[1] * fr)
        return np.real(output)
    
    def fi(k_parallel):
        return f(1j * k_parallel)
    
    if abs(phcs.ep[1] - phcs.ep[0]) * min([1 - fr, fr]) < 0.2:
        real_k_parallel = find_real_roots_for_small_and_big_q(f, qa)
    else:
        real_k_parallel = find_real_roots(f, nmax * k0a + 0.12)
    real_k_parallel = find_real_roots(f, nmax * k0a + 0.12)
    nreal = len(real_k_parallel)
    
    if nreal < nmode:
        nimag = nmode - nreal
        imag_k_parallel = 1j * np.array(
                find_n_roots_for_small_and_big_q(fi, qa, nimag))
        return [real_k_parallel, imag_k_parallel.tolist()]
    else:
        return [real_k_parallel, []]
    

def find_eigen_kpar_in_an_area(phcs, qa, k0a, num,
                               kpara_real_extreme,
                               kpara_imag_extreme,
                               mode="E"):
    """
    The eigenstates of the 1D photonic crystal.

    Paramters
    ---------
    phcs: PhotonicCrystalSlab
        the Photonic Crystal Slab which is a kind of class.
    qa: float
        the Bloch wave number
    k0a: float
        the frequency divided by (2pi*c)
    num: EssentialNumber
    kpara_real_extreme: list[float]
        there is a real eigenvalue between any the adjacent two in this list
    kpara_imag_extreme: list[float]
        there is an imaginary eigenvalue between any the adjacent two in this
        list
    mode: {"E", "H"}, optional
        the mode of the eigenstate
    
    Returns
    -------
    np.ndarray
        real k_parallels(eigenvalue) of eigenstates
    np.ndarray
        imaginary k_parallels(eigenvalue) of eigenstates
    """

    fr = phcs.fr
    
    if mode.lower() == "e":
        ep = phcs.ep
        mu = phcs.mu
    elif mode.lower() == "h":
        ep = -phcs.mu
        mu = -phcs.ep
        
    muepk0a = mu * ep * (k0a)**2
    mu0, mu1 = mu
    
    def f(kza):
        kya = np.sqrt(muepk0a - kza**2 + 0j)
        kya0, kya1 = kya
        eta = (kya1 * mu0) / (kya0 * mu1)
        output = np.cos(qa) - np.cos(kya0 * (1 - fr)) * np.cos(kya1 * fr) + 0.5 * (eta + 1 / eta) * np.sin(
            kya0 * (1 - fr)) * np.sin(kya1 * fr)
        return output.real

    def fi(kza):
        return f(1j * kza)
    
    n_real = 0
    n_image = 0
    real_k_parallel = []
    imag_k_parallel = []
    
    # find the Bloch waves' kz(real)
    b = 0
    fb = f(0)
    for tem_kpara in kpara_real_extreme:
        a = b
        b = tem_kpara
        fa = fb
        fb = f(b)
        if (fa*fb) < 0:
            real_k_parallel.append(dichotomy(f, a, b))
            n_real = n_real + 1
        if n_real == num.real:
            break
    # find the Bloch waves' kz(image)
    b = 0
    fb = fi(0)
    for tem_kpara in kpara_imag_extreme:
        a = b
        b = tem_kpara
        fa = fb
        fb = fi(b)
        if (fa*fb) < 0:
            imag_k_parallel.append(1j*dichotomy(fi, a, b))
            n_image = n_image + 1
        if n_image == num.imag:
            break
    
    return real_k_parallel, imag_k_parallel
