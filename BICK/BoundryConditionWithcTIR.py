# -*- coding: utf-8 -*-
import numpy as np

def getcoefficents(fields, nne, npo, constant_number=0):

    boundrymode = 2
    if boundrymode == 2:
        return getcoe2(fields, nne, npo, constant_number=0)
    

def getcoe1(fields, nne, npo, constant_number=0):
    
    real_fields = fields[0] * 1
    real_fields.extend(fields[1])
    imag_fields_in = fields[2] * 1
    imag_fields_re = fields[3] * 1
    
    field_components = ["Ex", "Ey", "Hx", "Hy"]
    extend_Matrix = []
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    nd = nne + npo + 1
    n_real_total = len(real_fields)
    n_imag_in = len(imag_fields_in)
    flag = 0
    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        for component in field_components:
            one_row = []
            inside_field_real_part = [field.fieldfc(i, component) 
                                      for field in real_fields]
            inside_field_imag_part = []
            for j in range(n_imag_in):
                field_mode = imag_fields_in[j].mode
                in_field = imag_fields_in[j].fieldfc(i, component)
                re_field = imag_fields_re[j].fieldfc(i, component)
                if field_mode == "E":
                    inside_field_imag_part.append(in_field + re_field)
                else:
                    inside_field_imag_part.append(in_field - re_field)
                
         
            outside_fields_tx = [0 for j in range(nd - 1)]
            outside_fields_ty = [0 for j in range(nd - 1)]
            
            if i != 0:
                expz = np.exp(1j * kzaouti * h / 2)
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
            
            one_row.extend(inside_field_real_part)
            one_row.extend(inside_field_imag_part)
            one_row.extend(outside_fields_tx)
            one_row.extend(outside_fields_ty)
            extend_Matrix.append(one_row)
        if i != 0:
            flag = flag + 1
    
    extend_Matrix = np.array(extend_Matrix)
    coefficients_Matrix = np.delete(extend_Matrix, 
                                    constant_number, 
                                    axis=1)
    constant_vector = - extend_Matrix[:, constant_number] * 1
    solve_coefficents = np.linalg.solve(coefficients_Matrix, constant_vector)
    
    coefs = [1] 
    coefs.extend(solve_coefficents[0:n_real_total + n_imag_in - 1])
    for j in range(n_imag_in):
        field_mode = imag_fields_in[j].mode
        if field_mode == "E":
            coefs.append(solve_coefficents[n_real_total - 1 + j])
        else:
            coefs.append(-solve_coefficents[n_real_total - 1 + j])
    
    tx = solve_coefficents[n_real_total + n_imag_in - 1:
                           n_real_total + n_imag_in - 1 + nd - 1]
    
    ty = solve_coefficents[n_real_total + n_imag_in - 1 + nd - 1:]
    
    return coefs, tx, ty

def getcoe2(fields, nne, npo, constant_number=0):
    
    real_fields = fields[0] * 1
    real_fields.extend(fields[1])
    imag_fields_in = fields[2] * 1
    imag_fields_re = fields[3] * 1
    
    field_mode = real_fields[0].mode
    if field_mode == "E":
        field_components = ["Ey", "Hx"]
    else:
        field_components = ["Hy", "Ex"]
    extend_Matrix = []
    qa = real_fields[0].qa
    k0a = real_fields[0].k0a
    kya = real_fields[0].kya
    h = real_fields[0].es.phcs.h / real_fields[0].es.phcs.a
    nd = nne + npo + 1
    
    n_imag_in = len(imag_fields_in)
    flag = 0
    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        for component in field_components:
            one_row = []
            inside_field_real_part = [field.fieldfc(i, component) 
                                      for field in real_fields]
            inside_field_imag_part = [imag_fields_in[j].fieldfc(i, component) +
                                      imag_fields_re[j].fieldfc(i, component)
                                      for j in range(n_imag_in)]
         
            outside_fields_t = [0 for j in range(nd - 1)]
            
            if i != 0:
                expz = np.exp(1j * kzaouti * h / 2)
                if component == "Ey" or component == "Hy":
                    outside_fields_t[flag] = -expz
                else:
                    if component == "Ex":
                        outside_fields_t[flag] = -kzaouti / k0a * expz
                    else:
                        outside_fields_t[flag] = kzaouti / k0a * expz
                    
            one_row.extend(inside_field_real_part)
            one_row.extend(inside_field_imag_part)
            one_row.extend(outside_fields_t)
            extend_Matrix.append(one_row)
        if i != 0:
            flag = flag + 1
    
    extend_Matrix = np.array(extend_Matrix)
    coefficients_Matrix = np.delete(extend_Matrix, 
                                    constant_number, 
                                    axis=1)
    constant_vector = - extend_Matrix[:, constant_number] * 1
    solve_coefficents = np.linalg.solve(coefficients_Matrix, constant_vector)
    
    return solve_coefficents


