from bicky.eigenkpar import find_eigen_kpar_in_an_area, find_eigen_kpar
from bicky.photoniccrystalbandprojection import find_band_projection
from bicky.photoniccrystalbandprojection import mini_frequncy
from bicky.field import FieldsWithCTIRInArea, FieldsWithCTIRMix
import numpy as np

class FindBICs:
    """find BICs in q-k0 space with single polarization.
    
    Attributes
    ----------
    BIC_qs: list[list[float]]
        each item contains BICs' qs for corresponding thickness.
    BIC_k0s: list[list[float]]
        each item contains BICs' k0s for corresponding thickness.
    BIC_hs: list[float]
        the thickness of PhC slab where BICs exist.
    """
    def __init__(self, phcs, num, mode="E", Nq=250):
        """Initialize the class, create the gridding.
        
        Parameters
        ----------
        phcs: PhotonicCrystalSlab
            the Photonic Crystal Slab which is a kind of class.
        num: EssentialNumber
        
        mode: {"E", "H",}, optional
            considered mode 
        Nq: int, optional
            number which we divided half of the Brillouin into
        """
        self.phcs = phcs
        self.num = num
        self.mode = mode
        if type(Nq) != int:
            raise ValueError("""Nq should be int.
                             """)
        if mode.lower() == 'h' or mode.lower() == 'e':
            deltaq = 0.5/Nq
            k0_floor, k0_ceiling, dataq, band_proj = \
                find_band_projection(phcs, num, mode=mode, Nq=Nq)
            datak0 = band_proj["k0a"]
            kpara_real_range_origin = band_proj["real"]
            kpara_imag_range_origin = band_proj["imag"]
            
            #gridding
            real_k_parallel, imag_k_parallel, qk0 = [], [], []
            for i in range(len(dataq)):
                qa = i * deltaq + dataq[0]
                # this is the area in which Bloch waves' kz(real and image) will be found
                kpara_real_range = []
                kpara_imag_range = []
                # this is the range of considered frequency 
                k0_range = []
                
                for j in range(len(datak0)):
                    datak0j = datak0[j]
                    if k0_floor[i] <= datak0j <= k0_ceiling[i]:
                        k0_range.append(datak0j)
                        kpara_real_range.append(
                            kpara_real_range_origin[j])
                        kpara_imag_range.append(
                            kpara_imag_range_origin[j])
                # compute the data     
                for k in range(len(k0_range)):
                    k0a = k0_range[k]
                    kpara_real_extreme = kpara_real_range[k]
                    kpara_imag_extreme = kpara_imag_range[k]
                    tem_real_k_parallel, tem_imag_k_parallel = \
                        find_eigen_kpar_in_an_area(phcs, qa*2*np.pi,
                                                   k0a*2*np.pi, num,
                                                   kpara_real_extreme,
                                                   kpara_imag_extreme,
                                                   mode=mode)
                    real_k_parallel.append(tem_real_k_parallel)
                    imag_k_parallel.append(tem_imag_k_parallel)
                    qk0.append([qa, k0a])
                  
            self.qk0 = qk0
            self.real_k_parallel = real_k_parallel
            self.imag_k_parallel = imag_k_parallel
        else:
            raise ValueError("""mode should only be 'E' or 'H'
                             """)
    
    
    def find_kpar(self, mode, Nq):
        phcs = self.phcs
        num = self.num
        deltaq = 0.5/Nq
        k0_floor, k0_ceiling, dataq, band_proj = \
            find_band_projection(phcs, num, mode=mode, Nq=Nq)
        datak0 = band_proj["k0a"]
        kpara_real_range_origin = band_proj["real"]
        kpara_imag_range_origin = band_proj["imag"]
        
        #gridding
        real_k_parallel, imag_k_parallel, qk0 = [], [], []
        for i in range(len(dataq)):
            qa = i * deltaq + dataq[0]
            # this is the area in which Bloch waves' kz(real and image) will be found
            kpara_real_range = []
            kpara_imag_range = []
            # this is the range of considered frequency 
            k0_range = []
            
            for j in range(len(datak0)):
                datak0j = datak0[j]
                if k0_floor[i] <= datak0j <= k0_ceiling[i]:
                    k0_range.append(datak0j)
                    kpara_real_range.append(
                        kpara_real_range_origin[j])
                    kpara_imag_range.append(
                        kpara_imag_range_origin[j])
            # compute the data     
            for k in range(len(k0_range)):
                k0a = k0_range[k]
                kpara_real_extreme = kpara_real_range[k]
                kpara_imag_extreme = kpara_imag_range[k]
                tem_real_k_parallel, tem_imag_k_parallel = \
                    find_eigen_kpar_in_an_area(phcs, qa*2*np.pi,
                                               k0a*2*np.pi, num,
                                               kpara_real_extreme,
                                               kpara_imag_extreme,
                                               mode=mode)
                real_k_parallel.append(tem_real_k_parallel)
                imag_k_parallel.append(tem_imag_k_parallel)
                qk0.append([qa, k0a])
              
        return qk0, real_k_parallel, imag_k_parallel
    
    
    def getcoeffs(self):
        """get the ratio of coefficients of two Bloch waves in opposite
        direction.
        """
        qk0 = self.qk0
        phcs, num = self.phcs, self.num
        real_k_parallel, imag_k_parallel = \
            self.real_k_parallel, self.imag_k_parallel
        kya = 0
        odd_coefs, even_coefs, kzas = [], [], []
        
        for i in range(len(qk0)):
            qa, k0a = qk0[i]
            temfield = FieldsWithCTIRInArea(phcs, num,
                                            k0a*2*np.pi,
                                            qa*2*np.pi, kya,
                                            real_k_parallel[i],
                                            imag_k_parallel[i],
                                            mode=self.mode)
            odd_coefs.append(temfield.odd_coefs_inside)
            even_coefs.append(temfield.even_coefs_inside)
            kzas.append(temfield.realkzs)
            
            
        self.odd_coefs = np.array(odd_coefs)
        self.even_coefs = np.array(even_coefs)
        self.kzas = np.array(kzas)
        
    
    def run(self, hstart, hend, Nh=20, limit=0.999):
        """search BICs by varying thickness of PhC slab.
        
        Parameters
        ----------
        hstart: float
            start searching in this thickness
        hend: float
            end searching in this thickness
        Nh: int, optional
            number of searching thickness
        limit:float
            the precision of judging if a point in q-k0 space is a BIC
        """
        qk0 = self.qk0
        num = self.num
        odd_coefs = self.odd_coefs
        even_coefs = self.even_coefs
        kzas = self.kzas
        n_real = num.real

        def find_bic(h):
            """
            This is a function to find bics in PhCS
            
            Parameters
            ----------
            h: int
                the thickness of PhCS
            
            Returns
            -------
            list[float]:
                the BICs' q
            list[float]:
                the BICs' k0
            """
            test = []
            odd_coefs_boundry = np.real(odd_coefs *
                                        np.exp(-1j * h * kzas)).tolist()
            
            even_coefs_boundry = np.real(even_coefs *
                                         np.exp(-1j * h * kzas)).tolist()

            for i in range(len(qk0)):
                neflag = n_real
                noflag = n_real
                sum_odd_real = 0
                sum_even_real = 0
                for j in range(n_real):
                    oddreal = odd_coefs_boundry[i][j]
                    evenreal = even_coefs_boundry[i][j]
                    if (-2+limit<oddreal<-limit):
                        noflag = noflag - 1
                    if (2-limit>evenreal>limit):
                        neflag = neflag - 1
                    sum_even_real = sum_even_real + evenreal
                    sum_odd_real = sum_odd_real + oddreal
                dataq = qk0[i][0]
                datak0 = qk0[i][1]
                if neflag == 0:
                    test.append([dataq, datak0, sum_even_real])
                if noflag == 0:
                    test.append([dataq, datak0, -sum_odd_real])
                
            bicregion = [[test[0]]]
            flag = 1
            limitdelta = 0.02
            
            for i in range(len(test) - 1):
                for j in range(len(bicregion)):
                    if (abs(bicregion[j][-1][0] - test[i+1][0])<limitdelta
                        and abs(bicregion[j][-1][1] - test[i+1][1])<limitdelta):
                        bicregion[j].append(test[i+1])
                        flag = 0
                        break
                    flag = 1
               
                if flag:
                    bicregion.append([test[i+1]])
            
            bic_q = []
            bic_k0 = []
            for onebicregion in bicregion:
                onebic = [onebicregion[0][0], onebicregion[0][1], onebicregion[0][2]]
                for j in range(len(onebicregion)-1):
                    if onebicregion[j+1][-1]>onebic[-1]:
                        onebic = [onebicregion[j+1][0], onebicregion[j+1][1], onebicregion[j+1][2]]
                bic_q.append(onebic[0])
                bic_k0.append(onebic[1])
            return bic_q, bic_k0
        
        rangeh = np.linspace(hstart, hend, Nh)
        
        bic_qs, bic_k0s, bic_hs = [], [], []
        for h in rangeh:
            try:
                bic_q, bic_k0 = find_bic(h)
                if bic_q:
                    bic_qs.append(bic_q)
                    bic_k0s.append(bic_k0)
                    bic_hs.append(h)
            except:
                pass
        self.bic_qs = bic_qs
        self.bic_k0s = bic_k0s
        self.bic_hs = bic_hs


class FindBICsMix:
    """find BICs in ky-k0 space with mix polarization.
    
    Attributes
    ----------
    BIC_kys: list[list[float]]
        each item contains BICs' qs for corresponding thickness.
    BIC_k0s: list[list[float]]
        each item contains BICs' k0s for corresponding thickness.
    BIC_hs: list[float]
        the thickness of PhC slab where BICs exist.
    """
    def __init__(self, phcs, num, qa, k0range=0.5*2*np.pi, Nk0=200):
        """Initialize the class, create the gridding.
        
        Parameters
        ----------
        phcs: PhotonicCrystalSlab

        num: EssentialNumber
        
        qa: float
            the Bloch wave number, in unit 1/a
        k0range: float, option
            the length of the range of k0, in unit 1/a
        Nk0: int, optional
            the number which we divided the k0range
        """
        self.phcs = phcs
        self.num = num
        delta = k0range/Nk0
        mink0 = mini_frequncy(phcs, num, qa, 0)
        maxk0 = mink0 + k0range
        kpe = []
        kph = []
        k0set = np.linspace(mink0, maxk0, num=Nk0)
        for k0s in k0set:    
            singe_kpe = find_eigen_kpar(phcs, k0s, qa, num.modes)
            singe_kph = find_eigen_kpar(phcs, k0s, qa, num.modes, mode="H")
            kpe.append(singe_kpe)
            kph.append(singe_kph)
        maxky = np.sqrt(maxk0**2 - qa**2)
        ky = 0
        
        
        def lightline(i, ky):
            """light line
            """
            if i%2:
                value_k0 = np.sqrt((qa/(2*np.pi) + (i - 1) / 2)**2 + ky**2)
            else:
                value_k0 = np.sqrt((i / 2 - qa/(2*np.pi))**2 + ky**2)
            return value_k0
        
        
        def num_large(array, value):
            """the number of items in the array which large than some value.
            """
            total = 0
            for item in array:
                if item>value:
                    total = total + 1
            return total
        kyk0 = []
        k_para_e = []
        k_para_h = []
        
        while ky<maxky:
            kys = ky/(2*np.pi)
            k0f = lightline(1, kys)*2*np.pi
            k0c = lightline(2, kys)*2*np.pi
            for i in range(len(k0set)): 
                k0 = k0set[i]
                kpei = kpe[i][0]
                kphi = kph[i][0]
                if k0>k0f and k0<k0c:
                    if num_large(kpei, ky)==2 and num_large(kphi, ky)==2:
                        kyk0.append([ky, k0])
                        if kpei[0]<ky and kphi[0]<ky:
                            imkpe = kpe[i][1]*1
                            imkph = kph[i][1]*1
                            imkpe.append(kpei[0])
                            imkph.append(kphi[0])
                            kppe = [kpei[1:], imkpe]
                            kpph = [kphi[1:], imkph]
                            
                            k_para_e.append(kppe)
                            k_para_h.append(kpph)
                        else:
                            k_para_e.append(kpe[i])
                            k_para_h.append(kph[i])
            ky = ky + delta
        self.kyk0 = kyk0
        self.k_para_e = k_para_e
        self.k_para_h = k_para_h
        
    
    
    def getcoeffs(self):
        """get the ratio of coefficients of two Bloch waves in opposite
        direction.
        """
        kyk0 = self.kyk0
        k_para_e = self.k_para_e
        k_para_h = self.k_para_h
        phcs = self.phcs
        num = self.num
        even_coefs = []
        odd_coefs = []
        kzas = []
        print(len(kyk0))
        for i in range(len(kyk0)):
            
            ky = kyk0[i][0]
            k0 = kyk0[i][1]
            kparae = k_para_e[i]
            kparah = k_para_h[i]
            f1 = FieldsWithCTIRMix(phcs, num, k0, 0, ky, kparae, kparah)
            even_coefs_inside = f1.even_coefs_inside
            odd_coefs_inside = f1.odd_coefs_inside
            realkzs = f1.realkzs
            even_coefs.append(even_coefs_inside[1:])
            odd_coefs.append(odd_coefs_inside[1:])
            kzas.append(realkzs[1:])
            
        self.kzas = np.array(kzas)
        self.even_coefs = np.array(even_coefs)
        self.odd_coefs = np.array(odd_coefs)

    def run(self, hstart, hend, Nh=20, limit=0.999):
        """search BICs by varying thickness of PhC slab.
        
        Parameters
        ----------
        hstart: float
            start searching in this thickness
        hend: float
            end searching in this thickness
        Nh: int, optional
            number of searching thickness
        limit:float
            the precision of judging if a point in q-k0 space is a BIC
        """
        kyk0 = self.kyk0
        num = self.num
        odd_coefs = self.odd_coefs
        even_coefs = self.even_coefs
        kzas = self.kzas
        n_real = num.real

        def find_bic(h):
            """
            This is a function to find bics in PhCS
            
            Parameters
            ----------
            h: int
                the thickness of PhCS
            
            Returns
            -------
            list[float]:
                the BICs' ky
            list[float]:
                the BICs' k0
            """
            test = []
            odd_coefs_boundry = np.real(odd_coefs *
                                        np.exp(-1j * h * kzas)).tolist()
            
            even_coefs_boundry = np.real(even_coefs *
                                         np.exp(-1j * h * kzas)).tolist()

            for i in range(len(kyk0)):
                neflag = n_real
                noflag = n_real
                sum_odd_real = 0
                sum_even_real = 0
                for j in range(n_real):
                    oddreal = odd_coefs_boundry[i][j]
                    evenreal = even_coefs_boundry[i][j]
                    if (-2+limit<evenreal<-limit):
                        neflag = neflag - 1
                    if (2-limit>oddreal>limit):
                        noflag = noflag - 1
                    sum_even_real = sum_even_real + evenreal
                    sum_odd_real = sum_odd_real + oddreal
                dataky = kyk0[i][0]
                datak0 = kyk0[i][1]
                if neflag == 0:
                    test.append([dataky, datak0, -sum_even_real])
                if noflag == 0:
                    test.append([dataky, datak0, sum_odd_real])
                
            bicregion = [[test[0]]]
            flag = 1
            limitdelta = 0.02
            
            for i in range(len(test) - 1):
                for j in range(len(bicregion)):
                    if (abs(bicregion[j][-1][0] - test[i+1][0])<limitdelta
                        and abs(bicregion[j][-1][1] - test[i+1][1])<limitdelta):
                        bicregion[j].append(test[i+1])
                        flag = 0
                        break
                    flag = 1
               
                if flag:
                    bicregion.append([test[i+1]])
            
            bic_ky = []
            bic_k0 = []
            for onebicregion in bicregion:
                onebic = [onebicregion[0][0], onebicregion[0][1], onebicregion[0][2]]
                for j in range(len(onebicregion)-1):
                    if onebicregion[j+1][-1]>onebic[-1]:
                        onebic = [onebicregion[j+1][0], onebicregion[j+1][1], onebicregion[j+1][2]]
                bic_ky.append(onebic[0]/(2*np.pi))
                bic_k0.append(onebic[1]/(2*np.pi))
            return bic_ky, bic_k0
        
        rangeh = np.linspace(hstart, hend, Nh)
        
        bic_kys, bic_k0s, bic_hs = [], [], []
        for h in rangeh:
            try:
                bic_ky, bic_k0 = find_bic(h)
                if bic_ky:
                    bic_kys.append(bic_ky)
                    bic_k0s.append(bic_k0)
                    bic_hs.append(h)
            except:
                pass
        self.bic_kys = bic_kys
        self.bic_k0s = bic_k0s
        self.bic_hs = bic_hs
        
     
        
