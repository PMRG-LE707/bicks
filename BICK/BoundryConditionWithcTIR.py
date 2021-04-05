# -*- coding: utf-8 -*-
import numpy as np

def getcoefficents(fields_upward, fields_downward, nne, npo, constant_number=0):
    extend_Matrix = []
    field_direction = ["Ex", "Ey", "Hx", "Hy"]
    qa = fields_upward[0].qa
    k0a = fields_upward[0].k0a
    kya = fields_upward[0].kya
    h = fields_upward[0].es.phcs.h / fields_upward[0].es.phcs.a
    nd = nne + npo + 1
    flag = 0
    for i in range(-nne, npo + 1, 1):
        kxai = i * 2 * np.pi + qa
        kzaouti = np.sqrt(k0a ** 2 - kxai ** 2 - kya ** 2 + 0j)
        for direction in field_direction:
            one_row = []
            inside_fields = [field.fieldfc(i, direction) for field in fields_upward[0:4]]
            inside_fields1 = [field.fieldfc(i, direction) for field in fields_downward[0:4]]        
            inside_fields2 = [fields_upward[j].fieldfc(i, direction) +
                              fields_downward[j].fieldfc(i, direction)  
                              for j in range(4, len(fields_upward))]
            outside_fields1 = [0 for j in range(nd - 1)]
            outside_fields2 = [0 for j in range(nd - 1)]
            if i != 0:
                expz = np.exp(1j * kzaouti * h / 2)
                if direction == "Ex":
                    outside_fields1[flag] = -expz
                
                elif direction == "Ey":
                    outside_fields2[flag] = -expz
                
                elif direction == "Hx":
                    outside_fields1[flag] = kxai * kya / (k0a * kzaouti) * expz
                    outside_fields2[flag] = (kya ** 2 / kzaouti + kzaouti) / k0a * expz 
                
                else:
                    outside_fields1[flag] = -(kxai ** 2 / kzaouti + kzaouti) / k0a * expz
                    outside_fields2[flag] = -kxai * kya / (k0a * kzaouti) * expz
            
            one_row.extend(inside_fields)
            one_row.extend(inside_fields1)
            one_row.extend(inside_fields2)
            
            one_row.extend(outside_fields1)
            one_row.extend(outside_fields2)               
            extend_Matrix.append(one_row)
        if i != 0:
            flag = flag + 1
    extend_Matrix = np.array(extend_Matrix)
    coefficients_Matrix = np.delete(extend_Matrix, 
                                    constant_number, 
                                    axis=1)
    print(coefficients_Matrix.shape)
    constant_vector = -extend_Matrix[:, constant_number] * 1
    print(np.linalg.det(coefficients_Matrix))
    return coefficients_Matrix
    solve_coefficents = np.linalg.solve(coefficients_Matrix, constant_vector)
    return solve_coefficents