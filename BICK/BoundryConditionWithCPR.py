import numpy as np


def coefficient_ratio(E_Fourier_coefs, H_Fourier_coefs, k0a, kzaout, kzam, kzae, kxa, n, phcs):
    """
    We will solve the linear equations wihch are derived from boundry condition and cTIR.
    So, we need the constant vetor and coefficient matrix
    --------------------
    Return: r1/a1(even), r2/a2(even), r1/a1(odd), r2/a2(odd) on the boundry
    """
    Ae = E_Fourier_coefs[:, 0, :]
    Be = E_Fourier_coefs[:, 1, :]
    Ce = E_Fourier_coefs[:, 2, :]
    Am = H_Fourier_coefs[:, 0, :]
    Bm = H_Fourier_coefs[:, 1, :]
    Cm = H_Fourier_coefs[:, 2, :]
    
    Ae = Ae.T
    Be = Be.T
    Ce = Ce.T
    Am = Am.T
    Bm = Bm.T
    Cm = Cm.T
    h = phcs.h
    constant_vectorEx = np.array([np.exp(1j * kzae[0] * h / 2) * Ae[i, 0] * kzae[0] * k0a for i in range(n.d)])
    constant_vectorHx = np.array([-kzaout[i] * (np.exp(1j * kzae[0] * h / 2) * Be[i, 0] * kxa) for i in range(n.d)])
    constant_vectorEy = np.array([0 for i in range(n.d)])
    constant_vectorHy = np.array([-kzaout[i] * np.exp(1j * kzae[0] * h / 2) * Ce[i, 0] * (kzae[0] * kzae[0] + kxa * kxa) for i in range(n.d)])
    constant_vector = np.concatenate((constant_vectorEx, constant_vectorHx, constant_vectorEy, constant_vectorHy))
    
    even_coefficient_matrix = []
    odd_coefficient_matrix = []
    
    # boundary condition Ex
    flag = 0
    ty = [0 for i in range(n.d - n.r)]
    for i in range(n.d):
        even_tem = []
        odd_tem = []
        ae = [(-np.exp(1j * kzae[j + 1] * h / 2) * Ae[i, j + 1] * kzae[j + 1] * k0a) for j in range(n.real - 1)]
        re = [(np.exp(-1j * kzae[j] * h / 2) * Ae[i, j] * kzae[j] * k0a) for j in range(n.real)]
        am = [(np.exp(1j * kzam[j] * h / 2) * Bm[i, j] * kxa) for j in range(n.real)]
        rm = [(np.exp(-1j * kzam[j] * h / 2) * Bm[i, j] * kxa) for j in range(n.real)]
        even_overlape = [(-np.exp(1j * kzae[j + n.real] * h / 2) + np.exp(-1j * kzae[j + n.real] * h / 2)) * Ae[i, j + n.real] * kzae[j + n.real] * k0a for j in range(n.overlap_e)]
        even_overlapm = [(np.exp(1j * kzam[j + n.real] * h / 2) - np.exp(-1j * kzam[j + n.real] * h / 2)) * Bm[i, j + n.real] * kxa for j in range(n.overlap_m)]
        odd_overlape = [(-np.exp(1j * kzae[j + n.real] * h / 2) - np.exp(-1j * kzae[j + n.real] * h / 2)) * Ae[i, j + n.real] * kzae[j + n.real] * k0a for j in range(n.overlap_e)]
        odd_overlapm = [(np.exp(1j * kzam[j + n.real] * h / 2) + np.exp(-1j * kzam[j + n.real] * h / 2)) * Bm[i, j + n.real] * kxa for j in range(n.overlap_m)]
        
        tx = [0 for j in range(n.d - n.r)]
        if i in n.list_r:
            "nothing"
        else:
            tx[flag] = -kxa * k0a * np.exp(1j * kzaout[i] * h / 2)
            flag = flag + 1
        
        ae.extend(re)
        ae.extend(am)     
        ae.extend(rm)
        ae.extend(tx)
        ae.extend(ty)
        even_tem.extend(ae)
        even_tem.extend(even_overlape)
        even_tem.extend(even_overlapm)
        
        odd_tem.extend(ae)
        odd_tem.extend(odd_overlape)
        odd_tem.extend(odd_overlapm)
        
        even_coefficient_matrix.append(even_tem)
        odd_coefficient_matrix.append(odd_tem)
    
    # boundary condition Hx
    flag = 0
    for i in range(n.d):
        kyaout = -2 * np.pi * (i - n.ne)
        even_tem = []
        odd_tem = []
        
        ae = [kzaout[i] * (np.exp(1j * kzae[j + 1] * h / 2) * Be[i, j + 1] * kxa) for j in range(n.real - 1)]
        re = [kzaout[i] * (np.exp(-1j * kzae[j] * h / 2) * Be[i, j] * kxa) for j in range(n.real)]
        am = [kzaout[i] * (-np.exp(1j * kzam[j] * h / 2) * Am[i, j] * kzam[j] * k0a) for j in range(n.real)]
        rm = [kzaout[i] * (np.exp(-1j * kzam[j] * h / 2) * Am[i, j] * kzam[j] * k0a) for j in range(n.real)]
        even_overlape = [kzaout[i] * (np.exp(1j * kzae[j + n.real] * h / 2) + np.exp(-1j * kzae[j + n.real] * h / 2)) * Be[i, j + n.real] * kxa for j in range(n.overlap_e)]
        even_overlapm = [kzaout[i] * (-np.exp(1j * kzam[j + n.real] * h / 2) - np.exp(-1j * kzam[j + n.real] * h / 2)) * Am[i, j + n.real] * kzam[j + n.real] * k0a for j in range(n.overlap_m)]
        odd_overlape = [kzaout[i] * (np.exp(1j * kzae[j + n.real] * h / 2) - np.exp(-1j * kzae[j + n.real] * h / 2)) * Be[i, j + n.real] * kxa for j in range(n.overlap_e)]
        odd_overlapm = [kzaout[i] * (-np.exp(1j * kzam[j + n.real] * h / 2) + np.exp(-1j * kzam[j + n.real] * h / 2)) * Am[i, j + n.real] * kzam[j + n.real] * k0a for j in range(n.overlap_m)]
        
        tx = [0 for j in range(n.d - n.r)]
        ty = [0 for j in range(n.d - n.r)]
        if i in n.list_r:
            "nothing"
        else:
            tx[flag] = kxa * kxa * kyaout * np.exp(1j * kzaout[i] * h / 2)
            ty[flag] = (kyaout * kyaout + kzaout[i] * kzaout[i]) * kxa * np.exp(1j * kzaout[i] * h / 2)
            flag = flag + 1

        
        ae.extend(re)
        ae.extend(am)
        ae.extend(rm)
        ae.extend(tx)
        ae.extend(ty)
        even_tem.extend(ae)
        even_tem.extend(even_overlape)
        even_tem.extend(even_overlapm)
        
        odd_tem.extend(ae)
        odd_tem.extend(odd_overlape)
        odd_tem.extend(odd_overlapm)
        
        even_coefficient_matrix.append(even_tem)
        odd_coefficient_matrix.append(odd_tem)
        
    # boundary condition Ey
    flag = 0
    tx = [0 for i in range(n.d - n.r)]
    for i in range(n.d):
        even_tem = []
        odd_tem = []
        ae = [0 for j in range(n.real - 1)]
        re = [0 for j in range(n.real)]
        am = [(np.exp(1j * kzam[j] * h / 2) * Cm[i, j] * (kzam[j] * kzam[j] + kxa * kxa)) for j in range(n.real)]
        rm = [(np.exp(-1j * kzam[j] * h / 2) * Cm[i, j] * (kzam[j] * kzam[j] + kxa * kxa)) for j in range(n.real)]
        even_overlape = [0 for j in range(n.overlap_e)]
        even_overlapm = [(np.exp(1j * kzam[j + n.real] * h / 2) - np.exp(-1j * kzam[j + n.real] * h / 2)) * Cm[i, j + n.real] * (kzam[j + n.real] * kzam[j + n.real] + kxa * kxa) for j in range(n.overlap_m)]
        odd_overlape = [0 for j in range(n.overlap_e)]
        odd_overlapm = [(np.exp(1j * kzam[j + n.real] * h / 2) + np.exp(-1j * kzam[j + n.real] * h / 2)) * Cm[i, j + n.real] * (kzam[j + n.real] * kzam[j + n.real] + kxa * kxa) for j in range(n.overlap_m)]
        
        ty = [0 for j in range(n.d - n.r)]
        if i in n.list_r:
            "nothing"
        else:
            ty[flag] = kxa * k0a * np.exp(1j * kzaout[i] * h / 2)
            flag = flag + 1
        
        
        ae.extend(re)
        ae.extend(am)
        ae.extend(rm)
        ae.extend(tx)
        ae.extend(ty)
        even_tem.extend(ae)
        even_tem.extend(even_overlape)
        even_tem.extend(even_overlapm)
        
        odd_tem.extend(ae)
        odd_tem.extend(odd_overlape)
        odd_tem.extend(odd_overlapm)
        
        even_coefficient_matrix.append(even_tem)
        odd_coefficient_matrix.append(odd_tem)
    
    # boundary condition Hy
    flag = 0
    for i in range(n.d):
        kyaout = -2 * np.pi * (i - n.ne)
        even_tem = []
        odd_tem = []
        
        ae = [kzaout[i] * (np.exp(1j * kzae[j + 1] * h / 2) * Ce[i, j + 1] * (kzae[j + 1] * kzae[j + 1] + kxa * kxa)) for j in range(n.real - 1)]
        re = [kzaout[i] * (np.exp(-1j * kzae[j] * h / 2) * Ce[i, j] * (kzae[j] * kzae[j] + kxa * kxa)) for j in range(n.real)]
        am = [0 for j in range(n.real)]
        rm = [0 for j in range(n.real)]
        
        even_overlape = [kzaout[i] * (np.exp(1j * kzae[j + n.real] * h / 2) + np.exp(-1j * kzae[j + n.real] * h / 2)) * Ce[i, j + n.real] * (kzae[j + n.real] * kzae[j + n.real] + kxa * kxa) for j in range(n.overlap_e)]
        even_overlapm = [0 for j in range(n.overlap_m)]
        odd_overlape = [kzaout[i] * (np.exp(1j * kzae[j + n.real] * h / 2) - np.exp(-1j * kzae[j + n.real] * h / 2)) * Ce[i, j + n.real] * (kzae[j + n.real] * kzae[j + n.real] + kxa * kxa) for j in range(n.overlap_e)]
        odd_overlapm = [0 for j in range(n.overlap_m)]
        
        tx = [0 for j in range(n.d - n.r)]
        ty = [0 for j in range(n.d - n.r)]
        if i in n.list_r:
            "nothing"
        else:
            tx[flag] = (kxa * kxa + kzaout[i] * kzaout[i]) * kxa * np.exp(1j * kzaout[i] * h / 2)
            ty[flag] = kxa * kyaout * kxa * np.exp(1j * kzaout[i] * h / 2)
            flag = flag + 1
        
        ae.extend(re)
        ae.extend(am)
        ae.extend(rm)
        ae.extend(tx)
        ae.extend(ty)
        even_tem.extend(ae)
        even_tem.extend(even_overlape)
        even_tem.extend(even_overlapm)
        
        odd_tem.extend(ae)
        odd_tem.extend(odd_overlape)
        odd_tem.extend(odd_overlapm)
        
        even_coefficient_matrix.append(even_tem)
        odd_coefficient_matrix.append(odd_tem)
        
    even_coefficient_matrix = np.array(even_coefficient_matrix)
    odd_coefficient_matrix = np.array(odd_coefficient_matrix)
    # coefficients
    c_even = np.linalg.solve(even_coefficient_matrix, constant_vector)  
    c_odd = np.linalg.solve(odd_coefficient_matrix, constant_vector)
    return c_even, c_odd