# -*- coding: utf-8 -*-
import numpy as np
import time

def getcoefficents(real_fields, imag_fields,
                   num, polarizationmode="single"):
    """
    I'am a function, I can give you the coefficents of
    different eigenstates(both incidence and reflection)
    when the cTIR of Bloch waves in one boundry happens.
    
    Paramters
    ----------
    :fields: a list of lenth 4, it contains incident and
        reflected fields with real kz and imag kz, respectively.
    :nne: negative diffraction oders.
    :npo: positive diffraction oders.
    :polarizationmode: "mix" or "single", if the mode is H 
        or E mode only, key the "single", if both, key the "mix"
    :return: the coefficents of different eigenstates in two
        kinds(even or odd for E mode)
    """
    
    if polarizationmode == "mix":
        t1 = time.time()
        for i in range(10 ** 3):
            getcoemix(real_fields, imag_fields, num)
        t2= time.time()
        print(t2 - t1)
        return getcoemix(real_fields, imag_fields, num)
    elif polarizationmode == "single":
        return getcoesingle(real_fields, imag_fields, num)
    else:
        "happy"
    

def getcoemix(real_fields, imag_fields, num, constant_number=0):
    """
    For the mix mode(both E and H mode)
    
    Paramters
    ----------
    :fields: a list of lenth 4, it contains incident and
        reflected fields with real kz and imag kz, respectively.
    :nne: negative diffraction oders
    :npo: positive diffraction oders
    :constant_number: the serial number of columm which is 
        constant in eqs.
    :return: the coefficents of different eigenstates in two
        kinds(even or odd for E mode)
    """
    
    nd = num.d
    nne = num.ne
    npo = num.po
    # fields in real and imag part
    field_components = ["Ex", "Ey", "Hx", "Hy"]
    even_extend_Matrix = []
    odd_extend_Matrix = []
    
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
    
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    
    # diffraction number
    n_real = len(real_fields)
    n_imag = len(imag_fields)
    
    # flag will growth 1 if it not in channel oder
    flag = 0
    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        expz = np.exp(1j * kzaouti * h / 2)
        [field.fieldfc(i, field_components) for field in real_fields]
        [field.fieldfc(i, field_components) for field in imag_fields]
        for component in field_components:
            # one row in extended matrix
            even_one_row = []
            odd_one_row = []
            inside_field_real_part = np.array([field.field_Fourier[component] 
                                      for field in real_fields]).flatten().tolist()
            
            odd_inside_field_imag_part = []
            even_inside_field_imag_part = []
            for j in range(n_imag):
                field_mode = imag_fields[j].mode
                fieldfcs = imag_fields[j].field_Fourier[component] 
                in_field = fieldfcs[0]
                re_field = fieldfcs[1]
                
                if field_mode == "E":
                    odd_inside_field_imag_part.append(in_field - re_field)
                    even_inside_field_imag_part.append(in_field + re_field)
                else:
                    odd_inside_field_imag_part.append(in_field + re_field)
                    even_inside_field_imag_part.append(in_field - re_field)
                
            outside_fields_tx = [0 for j in range(nd - 1)]
            outside_fields_ty = [0 for j in range(nd - 1)]
            
            if i != 0:
                if component == "Ex":
                    outside_fields_tx[flag] = -expz
                
                elif component == "Ey":
                    outside_fields_ty[flag] = -expz
                
                elif component == "Hx":
                    outside_fields_tx[flag] = kxai * kya / (k0a * kzaouti) * expz
                    outside_fields_ty[flag] = (kya ** 2 / kzaouti + kzaouti) / k0a * expz 
                
                else:
                    outside_fields_tx[flag] = -(kxai ** 2 / kzaouti + kzaouti) / k0a * expz
                    outside_fields_ty[flag] = -kxai * kya / (k0a * kzaouti) * expz
            
            even_one_row.extend(inside_field_real_part)
            even_one_row.extend(even_inside_field_imag_part)
            even_one_row.extend(outside_fields_tx)
            even_one_row.extend(outside_fields_ty)
            
            odd_one_row.extend(inside_field_real_part)
            odd_one_row.extend(odd_inside_field_imag_part)
            odd_one_row.extend(outside_fields_tx)
            odd_one_row.extend(outside_fields_ty)
            
            even_extend_Matrix.append(even_one_row)
            odd_extend_Matrix.append(odd_one_row)
        if i != 0:
            flag = flag + 1
    
    n_real *= 2
    def solve(extend_Matrix):
        """
        Give the extended matrix to get the solution.
        
        Paramters
        ---------------
        :extend_Matrix: extend matrix provided by the eqs
        :return: the coefficients of the fields in the PhCs and
            tx and ty
        """
        
        extend_Matrix = np.array(extend_Matrix)
        
        coefficients_Matrix = np.delete(extend_Matrix, 
                                        constant_number, 
                                        axis=1)
        constant_vector = - extend_Matrix[:, constant_number] * 1
        solve_coefficents = np.linalg.solve(coefficients_Matrix, constant_vector)
        
        coefs = [1]
        
        coefs.extend(solve_coefficents[0:n_real + n_imag - 1])
        for j in range(n_imag):
            field_mode = imag_fields[j].mode
            if field_mode == "E":
                coefs.append(solve_coefficents[n_real - 1 + j])
            else:
                coefs.append(-solve_coefficents[n_real - 1 + j])
        
        tx = solve_coefficents[n_real + n_imag - 1:
                               n_real + n_imag - 1 + nd - 1]
        
        ty = solve_coefficents[n_real + n_imag - 1 + nd - 1:]

        return (coefs, tx, ty)     
    
    return solve(even_extend_Matrix), solve(odd_extend_Matrix)


def getcoesingle(real_fields, imag_fields, num, constant_number=0):
    """
    For the single mode(only E or H mode)
    
    Paramters
    ----------
    :fields: a list of lenth 4, it contains incident and
        reflected fields with real kz and imag kz, respectively.
    :nne: negative diffraction oders
    :npo: positive diffraction oders
    :constant_number: the serial number of columm which is 
        constant in eqs.
    :return: the coefficents of different eigenstates in two
        kinds(even or odd)
    """
    nd = num.d
    nne = num.ne
    npo = num.po
    
    # fields in real and imag part
    
    field_mode = real_fields[0].mode
    if field_mode == "E":
        field_components = ["Ey", "Hx"]
    else:
        field_components = ["Hy", "Ex"]
    even_extend_Matrix = []
    odd_extend_Matrix = []
    
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    
    n_imag = num.imag
    n_real = num.real
    flag = 0
    expzh = np.exp(1j*h*np.array([field.kza[0]
                                  for field in real_fields]))
    
    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        expz = np.exp(1j * kzaouti * h / 2)
        [field.fieldfc(i, field_components) for field in real_fields]
        [field.fieldfc(i, field_components) for field in imag_fields]
        for component in field_components:
            even_one_row = []
            odd_one_row = []
            inside_field_real_part = np.array([field.field_Fourier[component] 
                                      for field in real_fields]).flatten().tolist()
            
            odd_inside_field_imag_part = []
            even_inside_field_imag_part = []
            for j in range(n_imag):
                fieldfcs = imag_fields[j].field_Fourier[component] 
                in_field = fieldfcs[0]
                re_field = fieldfcs[1]
                
                if field_mode == "E":
                    odd_inside_field_imag_part.append(in_field - re_field)
                    even_inside_field_imag_part.append(in_field + re_field)
                else:
                    odd_inside_field_imag_part.append(in_field + re_field)
                    even_inside_field_imag_part.append(in_field - re_field)
                
         
            outside_fields_t = [0 for j in range(nd - num.r)]
            
            if i not in num.listr:
                if component == "Ey" or component == "Hy":
                    outside_fields_t[flag] = -expz
                else:
                    if component == "Ex":
                        outside_fields_t[flag] = -kzaouti / k0a * expz
                    else:
                        outside_fields_t[flag] = kzaouti / k0a * expz
                    
            even_one_row.extend(inside_field_real_part)
            even_one_row.extend(even_inside_field_imag_part)
            even_one_row.extend(outside_fields_t)
            
            odd_one_row.extend(inside_field_real_part)
            odd_one_row.extend(odd_inside_field_imag_part)
            odd_one_row.extend(outside_fields_t)
            
            even_extend_Matrix.append(even_one_row)
            odd_extend_Matrix.append(odd_one_row)
        if i not in num.listr:
            flag = flag + 1
    
    def solve(extend_Matrix):
        """
        Give the extended matrix to get the solution.
        
        Paramters
        ---------------
        :extend_Matrix: extend matrix provided by the eqs
        :return: the coefficients of the fields in the PhCs and
            tx and ty
        """
        
        extend_Matrix = np.array(extend_Matrix)
        coefficients_Matrix = np.delete(extend_Matrix, 
                                        constant_number, 
                                        axis=1)
        constant_vector = - extend_Matrix[:, constant_number] * 1
        solve_coefficents = np.linalg.solve(coefficients_Matrix,
                                            constant_vector)
        coefficents = np.append(np.ones(1, dtype=complex),
                               solve_coefficents)
        
        real_coeffs_ratio = [coefficents[i] / coefficents[i+1] 
                             for i in range(0, 2*n_real, 2)]

        return real_coeffs_ratio# * expzh
    
    return solve(even_extend_Matrix), solve(odd_extend_Matrix)


