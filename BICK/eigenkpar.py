import numpy as np
from MathTool import find_real_roots_for_small_and_big_q, find_n_roots_for_small_and_big_q, find_real_roots

def find_eigen_kpar(phcs, k0a, qa, nmode, mode="E"):
    """
    The eigenstates of the 1D photonic crystal.

    :param phcs: the Photonic Crystal Slab which is a kind of class.
    :param q: the Bloch wave number
    :param k0: the frequency divided by (2pi*c)
    :param nimag: the number of considered evanescent modes
    :return: The Fourier coefficients of every Bloch waves' kz in PhC(all_kza)
    """

    fr = phcs.fr
    
    if mode == "E":
        ep = phcs.ep
        mu = phcs.mu
    else:
        ep = -np.array(phcs.mu)
        mu = -np.array(phcs.ep)
    
    n1 = np.sqrt(max(phcs.ep))
    
    def f(k_parallel):
        kya = [np.sqrt(mu[i] * ep[i] * (k0a) ** 2 - k_parallel ** 2 + 0j) for i in range(2)]
        eta = (kya[1] * mu[0]) / (kya[0] * mu[1])
        output = np.cos(qa) - np.cos(kya[0] * (1 - fr)) * np.cos(kya[1] * fr) + 0.5 * (eta + 1 / eta) * np.sin(
            kya[0] * (1 - fr)) * np.sin(kya[1] * fr)
        return output.real
    
    def fi(k_parallel):
        return f(1j * k_parallel)
    
    
    if abs(phcs.ep[1] - phcs.ep[0]) * min([1 - fr, fr]) < 0.2:
        real_k_parallel = find_real_roots_for_small_and_big_q(f, qa)
    else:
        real_k_parallel = find_real_roots(f, n1 * k0a + 0.12)
    
    nreal = len(real_k_parallel)
    if nreal < nmode:
        nimag = nmode - nreal
        imag_k_parallel = 1j * np.array(find_n_roots_for_small_and_big_q(fi, qa, nimag))
        return real_k_parallel, imag_k_parallel
    else:
        return real_k_parallel[0:nmode], []
    
    

