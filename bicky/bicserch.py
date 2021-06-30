from bicky.eigenkpar import find_eigen_kpar_in_an_area
from bicky.photoniccrystalbandprojection import find_band_projection
from bicky.field import FieldsWithCTIRInArea
import numpy as np

class FindBICs:
    """
    find BICs in q-k0 space
    """
    def __init__(self, phcs, num, mode="E", Nq=250):
        """
        Initialize the class, create the gridding.
        
        Paramters
        ---------
        phcs: PhotonicCrystalSlab
            the Photonic Crystal Slab which is a kind of class.
        
        num: EssentialNumber
            
        mode: {"E", "H"}, optional
            considered mode
            
        Nq: int, optional
            number which we divided half of Brillouin into
        
        """
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
            
        self.mode = mode    
        self.qk0 = qk0
        self.phcs = phcs
        self.num = num
        self.real_k_parallel = real_k_parallel
        self.imag_k_parallel = imag_k_parallel
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
        qk0 = self.qk0
        phcs, num = self.phcs, self.num
        real_k_parallel, imag_k_parallel = \
            self.real_k_parallel, self.imag_k_parallel
        kya = 0
        odd_ceofs, even_ceofs, kzas = [], [], []
        
        for i in range(len(qk0)):
            qa, k0a = qk0[i]
            temfield = FieldsWithCTIRInArea(phcs, num,
                                            k0a*2*np.pi,
                                            qa*2*np.pi, kya,
                                            real_k_parallel[i],
                                            imag_k_parallel[i],
                                            mode=self.mode)
            odd_ceofs.append(temfield.odd_coefs_inside)
            even_ceofs.append(temfield.even_coefs_inside)
            kzas.append(temfield.realkzs)
            
            
        self.odd_ceofs = np.array(odd_ceofs)
        self.even_ceofs = np.array(even_ceofs)
        self.kzas = np.array(kzas)
        
    def run(self, hstart, hend, Nh=20, limit=0.999):
        qk0 = self.qk0
        num = self.num
        odd_ceofs = self.odd_ceofs
        even_ceofs = self.even_ceofs
        kzas = self.kzas
        n_real = num.real

        def find_bic(h):
            """
            This is a function to find bics in PhCS
            --------------------------------
            h: int
                the thickness of PhCS
            return: the coordinate(q, k0) of bics
            """
            test = []
            odd_ceofs_boundry = np.real(odd_ceofs *
                                        np.exp(-1j * h * kzas)).tolist()
            
            even_ceofs_boundry = np.real(even_ceofs *
                                         np.exp(-1j * h * kzas)).tolist()

            for i in range(len(qk0)):
                neflag = n_real
                noflag = n_real
                sum_odd_real = 0
                sum_even_real = 0
                for j in range(n_real):
                    oddreal = odd_ceofs_boundry[i][j]
                    evenreal = even_ceofs_boundry[i][j]
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
        