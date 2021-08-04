# -*- coding: utf-8 -*-
import numpy as np

def getcoemix(real_fields, imag_fields, num, constant_number=2):
    """
    For the mix mode(both E and H mode)
    
    Parameters
    ----------
    real_fields: list[FieldInPhCS]
        the fields from eigenstates with real kz
    imag_fields: list[FieldInPhCS]
        the fields from eigenstates with imaginary kz
    num: EssentialNumber
    constant_number: int, optional
        the serial number of columm of extend matricx which represents
        constant in eqs.
    
    Returns
    -------
    list[float]
        the ratio of coefficients of two Bloch waves in opposite
        direction(the tangential compoments of E are even in z direction).
    list[float]
        the ratio of coefficients of two Bloch waves in opposite
        direction(the tangential compoments of E are odd in z direction).
    list[float]
        real kz of all Bloch waves
    """
    nd = num.d
    # fields in real and imag part
    field_components = ["Ex", "Ey", "Hx", "Hy"]
    even_extend_Matrix = []
    odd_extend_Matrix = []
    
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
    real_kzas = np.array([field.kza[0]
                          for field in real_fields])
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    n_real = len(real_fields)
    n_imag = len(imag_fields)
    
    # flag will growth 1 if it not in channel order
    flag = 0
    for i in range(-num.ne, num.po + 1, 1):
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
                                      for field in real_fields])\
                .flatten().tolist()
            
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
                
            outside_fields_tx = [0 for j in range(nd - len(num.listr))]
            outside_fields_ty = [0 for j in range(nd - len(num.listr))]
            
            if i not in num.listr:
                if component == "Ex":
                    outside_fields_tx[flag] = -expz
                
                elif component == "Ey":
                    outside_fields_ty[flag] = -expz
                
                elif component == "Hx":
                    outside_fields_tx[flag] = kxai * kya / (k0a * kzaouti)\
                        * expz
                    outside_fields_ty[flag] = (kya**2 / kzaouti + kzaouti)\
                        / k0a * expz 
                
                else:
                    outside_fields_tx[flag] = -(kxai**2 / kzaouti + kzaouti)\
                        / k0a * expz
                    outside_fields_ty[flag] = -kxai * kya / (k0a * kzaouti)\
                        * expz

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
        if i not in num.listr:
            flag = flag + 1
    
    def solve(extend_Matrix):
        """
        Give the extended matrix to get the solution.
        
        Parameters
        ----------
        extend_Matrix: np.ndarray(dtype=np.float)
            extend matrix provided by the eqs
        
        Returns
        -------
        list[float]
            the ratio of coefficients of two Bloch waves in opposite
            direction on the boundry.
        """
        
        extend_Matrix = np.array(extend_Matrix)
        coefficients_Matrix = np.delete(extend_Matrix, 
                                        constant_number, 
                                        axis=1)
        constant_vector = - extend_Matrix[:, constant_number] * 1
        solve_coefficents = np.linalg.solve(coefficients_Matrix,
                                            constant_vector)
        
        coefficents = np.insert(solve_coefficents,
                                constant_number,
                                1.0)
        real_coeffs_ratio = [coefficents[i] / coefficents[i+1]
                             for i in range(2, 2*n_real, 2)]

        return real_coeffs_ratio   
    
    return solve(even_extend_Matrix), solve(odd_extend_Matrix), real_kzas[1:]


def getcoesingle(real_fields, imag_fields, num, constant_number=0):
    """
    For the single mode(only E or H mode)
    
    Parameters
    ----------
    real_fields: list[FieldInPhCS]
        the fields from eigenstates with real kz
    imag_fields: list[FieldInPhCS]
        the fields from eigenstates with imaginary kz
    num: EssentialNumber
    constant_number: int, optional
        the serial number of columm of extend matricx which represents
        constant in eqs.
    
    Returns
    -------
    list[float]
        ratio of coefficients of two Bloch waves in opposite
        direction(the tangential compoments of E are even in z direction).
    list[float]
        ratio of coefficients of two Bloch waves in opposite
        direction(the tangential compoments of E are odd in z direction).
    list[float]
        real kz of all Bloch waves
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
    real_kzas = np.array([field.kza[0]
                          for field in real_fields])
    expzh = np.exp(1j*h*real_kzas)

    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        expz = np.exp(1j * kzaouti * h / 2)
        [field.fieldfc(i, field_components) for field in real_fields]
        [field.fieldfc(i, field_components) for field in imag_fields]
        for component in field_components:
            even_one_row = []
            odd_one_row = []
            inside_field_real_part = np.array([
                    field.field_Fourier[component] 
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
                
         
            outside_fields_t = [0 for j in range(nd - len(num.listr))]
            
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
        
        Parameters
        ----------
        extend_Matrix: np.ndarray(dtype=np.float)
            extend matrix provided by the eqs
        
        Returns
        -------
        list[float]
            the ratio of coefficients of two Bloch waves in opposite
            direction on the boundry.
        """
        
        extend_Matrix = np.array(extend_Matrix)
        coefficients_Matrix = np.delete(extend_Matrix, 
                                        constant_number, 
                                        axis=1)
        constant_vector = - extend_Matrix[:, constant_number] * 1
        solve_coefficents = np.linalg.solve(coefficients_Matrix,
                                            constant_vector)
        coefficents = np.insert(solve_coefficents,
                                constant_number,
                                1.0)
        real_coeffs_ratio = [coefficents[i] / coefficents[i+1] 
                             for i in range(0, 2*n_real, 2)]
        return real_coeffs_ratio * expzh
    
    return solve(even_extend_Matrix), solve(odd_extend_Matrix), real_kzas


def singleboundary(real_fields, imag_fields, num, constant_number=0):
    """
    compute the reflection coefficients on the single boundary(not the slab)
    
    Parameters
    ----------
    real_fields: list[FieldInPhCS]
        the fields from eigenstates with real kz
    imag_fields: list[FieldInPhCS]
        the fields from eigenstates with imaginary kz
    num: EssentialNumber
    constant_number: int, optional
        the serial number of columm of extend matricx which represents
        constant in eqs.
    
    Returns
    -------
    list[float]
        ratio of coefficients of two Bloch waves in opposite
        direction(the tangential compoments of E are even in z direction).
    """
    
    nd = num.d
    # fields in real and imag part
    field_components = ["Ex", "Ey", "Hx", "Hy"]
    extend_Matrix = []
    
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
  
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    
    # flag will growth 1 if it not in channel order
    flag = 0
    for i in range(-num.ne, num.po + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        expz = np.exp(1j * kzaouti * h / 2)
        [field.fieldfc(i, field_components) for field in real_fields]
        [field.fieldfc(i, field_components) for field in imag_fields]
        for component in field_components:
            # one row in extended matrix
            one_row = []
            inside_field_real_part = np.array([field.field_Fourier[component] 
                                      for field in real_fields])\
                .flatten().tolist()
            inside_field_imag_part = np.array([field.field_Fourier[component][1] 
                                      for field in imag_fields])
            
            
            outside_fields_tx = [0 for j in range(nd - 1)]
            outside_fields_ty = [0 for j in range(nd - 1)]
            
            if i not in num.listr:
                if component == "Ex":
                    outside_fields_tx[flag] = -expz
                
                elif component == "Ey":
                    outside_fields_ty[flag] = -expz
                
                elif component == "Hx":
                    outside_fields_tx[flag] = kxai * kya / (k0a * kzaouti)\
                        * expz
                    outside_fields_ty[flag] = (kya**2 / kzaouti + kzaouti)\
                        / k0a * expz 
                
                else:
                    outside_fields_tx[flag] = -(kxai**2 / kzaouti + kzaouti)\
                        / k0a * expz
                    outside_fields_ty[flag] = -kxai * kya / (k0a * kzaouti)\
                        * expz

            one_row.extend(inside_field_real_part)
            one_row.extend(inside_field_imag_part)
            one_row.extend(outside_fields_tx)
            one_row.extend(outside_fields_ty)

            extend_Matrix.append(one_row)
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

        return coefficents[1]   
    
    return solve(extend_Matrix)

    
