from eigenkpar import find_eigen_kpar_in_an_area
from PhotonicCrystalBandProjection import find_band_projection
from PhotonicCrystalSlab import PhotonicCrystalSlab, EssentialNumber
from Field import FieldsWithCTIRInAera
import numpy as np
import time
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
        t1 = time.time()
        k0_floor, k0_ceiling, dataq, band_proj = \
            find_band_projection(phcs, num, mode=mode, Nq=Nq)
        datak0 = band_proj["k0a"]
        kpara_real_range_origin = band_proj["real"]
        kpara_imag_range_origin = band_proj["imag"]
        t2 = time.time()  
        print(t2 - t1)
        
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
        self.phcs = phcs
        self.num = num
        self.real_k_parallel = real_k_parallel
        self.imag_k_parallel = imag_k_parallel
    
    def getcoeffs(self):
        qk0 = self.qk0
        phcs, num = self.phcs, self.num
        real_k_parallel, imag_k_parallel = \
            self.real_k_parallel, self.imag_k_parallel
        kya = 0
        ctirfields = []
        for i in range(len(qk0)):
            qa, k0a = qk0[i]
            ctirfields.append(FieldsWithCTIRInAera(phcs, num,
                                       k0a*2*np.pi,
                                       qa*2*np.pi, kya,
                                       real_k_parallel[i],
                                       imag_k_parallel[i]))
        self.ctirfields = ctirfields
    def run(self, hstart, hend, Nh=100):
        "sad"
    
    
a = 1
h = 1.4 * a
fr = 0.5
ep = np.array([1.0, 4.9])
phcs = PhotonicCrystalSlab(h, ep, fr, a)

num = EssentialNumber(n_radiation=3)
hstart = 1
hend = 3

fb1 = FindBICs(phcs, num)  
fb1.getcoeffs()
bics = fb1.run(hstart, hend)