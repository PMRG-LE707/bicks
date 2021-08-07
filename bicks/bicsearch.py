from bicks.eigenkpar import find_eigen_kpar_in_an_area, find_eigen_kpar
from bicks.photoniccrystalbandprojection import find_band_projection
from bicks.photoniccrystalbandprojection import mini_frequncy
from bicks.field import FieldsWithCTIRInArea, FieldsWithCTIRMix
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import time

class FindBICs:
    """find BICs in q-k0 space with single polarization.
    
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
            self.k0_floor = k0_floor
            self.k0_ceiling = k0_ceiling
            self.dataq = dataq
            datak0 = band_proj["k0a"]
            kpara_real_range_origin = band_proj["real"]
            kpara_imag_range_origin = band_proj["imag"]
            print("=============")
            print("Initializing:")
            start = time.time()
            #gridding
            real_k_parallel, imag_k_parallel, qk0 = [], [], []
            flagenum = len(dataq)//50
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
                    if len(tem_real_k_parallel) == num.real:
                        if len(tem_imag_k_parallel) == num.imag:
                            real_k_parallel.append(tem_real_k_parallel)
                            imag_k_parallel.append(tem_imag_k_parallel)
                            qk0.append([qa, k0a])
                if i%flagenum == 0:
                    iky = int(i/len(dataq)*50)+1
                    aii = "*" * iky
                    bii = "." * (50 - iky)
                    cii = iky / 50 * 100
                    dur = time.time() - start
                    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                          end = "")      
            print("\n" + "Initialization accomplished.")
            self.qk0 = qk0
            self.real_k_parallel = real_k_parallel
            self.imag_k_parallel = imag_k_parallel
        else:
            raise ValueError("""mode should only be 'E' or 'H'
                             """)
            
    
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
        print("=============")
        print("Computing:")
        start = time.time()
        flagenum = len(qk0)//50
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
            
            if i%flagenum == 0:
                iky = int(i/len(qk0)*50)+1
                aii = "*" * iky
                bii = "." * (50 - iky)
                cii = iky / 50 * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                      end = "")      
            
            
        self.odd_coefs = np.array(odd_coefs)
        self.even_coefs = np.array(even_coefs)
        self.kzas = np.array(kzas)
        print("\n" + "Computation accomplished.")
        
    
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
                        and \
                        abs(bicregion[j][-1][1] - test[i+1][1])<limitdelta):
                        bicregion[j].append(test[i+1])
                        flag = 0
                        break
                    flag = 1
               
                if flag:
                    bicregion.append([test[i+1]])
            
            bic_q = []
            bic_k0 = []
            for onebicregion in bicregion:
                onebic = [onebicregion[0][0],
                          onebicregion[0][1],
                          onebicregion[0][2]]
                for j in range(len(onebicregion)-1):
                    if onebicregion[j+1][-1]>onebic[-1]:
                        onebic = [onebicregion[j+1][0],
                                  onebicregion[j+1][1],
                                  onebicregion[j+1][2]]
                bic_q.append(onebic[0])
                bic_k0.append(onebic[1])
            return bic_q, bic_k0
        
        rangeh = np.linspace(hstart, hend, Nh)
        print("=============")
        print("Searching:")
        start = time.time()
        bic_qs, bic_k0s, bic_hs = [], [], []
        ikk=0
        flagenum = Nh//50
        nbics=0
        for h in rangeh:
            ikk = ikk+1
            if Nh >= 50:
                if ikk%flagenum == 0:
                    iky = int(ikk/Nh*50)
                    aii = "*" * iky
                    bii = "." * (50 - iky)
                    cii = iky / 50 * 100
                    dur = time.time() - start
                    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                          end = "") 
            else:
                iky = int(ikk/Nh*50)
                aii = "*" * iky
                bii = "." * (50 - iky)
                cii = iky / 50 * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                      end = "") 
            try:
                bic_q, bic_k0 = find_bic(h)
                if bic_q:
                    bic_qs.append(bic_q)
                    bic_k0s.append(bic_k0)
                    bic_hs.append(h)
                    nbics = len(bic_q) + nbics
            except:
                pass
        print("\n" + "Search accomplished.")
        print("Number of BICs found: ", nbics)
        self.bic_qs = bic_qs
        self.bic_k0s = bic_k0s
        self.bic_hs = bic_hs
        

    def showbic(self,i=0):
        """show bics in the k-omega space for one particular h.
        
        Parameters
        ----------
        i: int, optional
            the serial number of bic_hs
        """
        h = self.bic_hs
        bic_q = self.bic_qs
        bic_k0 = self.bic_k0s
        dataq = self.dataq
        k0_ceiling = self.k0_ceiling
        k0_floor = self.k0_floor
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('$k_x(2\pi/a)$')
        ax.set_ylabel('$\omega(2\pi c/a)$')
        if h == []:
            h = 0
        else:
            h = h[i]
        ax.set_title("BICs in $k_x-\omega$ space($h=%.3fa, k_y=0$)"%h)
        ax.plot(dataq, k0_ceiling, 'b', ls=':')
        ax.plot(dataq, k0_floor, 'black', ls='--')
        ax.fill_between(dataq, k0_ceiling, k0_floor,
                        color='C1', alpha=0.3,
                        interpolate=True,
                        label="Searching range")
        if bic_k0:
            ax.scatter(bic_q[i], bic_k0[i], marker='*',
                       s=100, c="red", edgecolors="black", 
                       label="BIC")
        plt.legend(markerscale=1)
        plt.show()
    
    
    def dynamicplot(self, save=False):
        """show bics in the k-omega space with variant h.
        
        Parameters
        ----------
        save: str, optional
            the path to save the dynamic picture.
        """
        bic_h = self.bic_hs
        bic_q = self.bic_qs
        bic_k0 = self.bic_k0s
        dataq = self.dataq
        k0_ceiling = self.k0_ceiling
        k0_floor = self.k0_floor
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('$k_x(2\pi/a)$')
        ax.set_ylabel('$\omega(2\pi c/a)$')
        ax.set_title("BICs in $k_x-\omega$ space($k_y=0$)")
        
        h_template = '$h$ = %.3f $a$'
        h_text = ax.text(0, np.min(k0_floor) + 0.02, '', fontsize=14,
                         bbox=dict(boxstyle="round4", fc="maroon", alpha=0.3))
        ax.plot(dataq, k0_ceiling, 'b', ls=':')
        ax.plot(dataq, k0_floor, 'black', ls='--')
        ax.fill_between(dataq, k0_ceiling, k0_floor,
                        color='lightskyblue', alpha=0.4,
                        interpolate=True,
                        label="Searching range")
        
        bics, = ax.plot([], [], 'o', marker='*',
                        markersize=10, color = 'red',
                        animated=True, label='BICs')
        
        def update(i):
            try:
                x = bic_q[i]
                y = bic_k0[i]
            except BaseException:
                x, y = [], []
            bics.set_data(x, y)
            h_text.set_text(h_template % bic_h[i])
            return bics, h_text
        
        
        anim = ani.FuncAnimation(
            fig,
            update,
            frames=np.arange(
                0,
                len(bic_h),
                1),
            interval=400,
            blit=True)
        plt.legend()
        plt.tight_layout()
        
        if save:
            anim.save(save, writer="imagemagick")
     
    
class FindBICsMix:
    """find BICs in ky-k0 space with mix polarization.
    
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
        self.qa = qa
        self.k0range = k0range
        self.Nk0 = Nk0
        
        delta = k0range/Nk0
        mink0 = mini_frequncy(phcs, num, qa, 0)
        maxk0 = mink0 + k0range
        kpe = []
        kph = []
        k0set = np.linspace(mink0, maxk0, num=Nk0)
        nik0 = 0
        start = time.time()
        print("=================")
        print("Initializing:")
        flagenum = Nk0//50
        for k0s in k0set:    
            nik0 = nik0 + 1
            singe_kpe = find_eigen_kpar(phcs, k0s, qa, num.modes)
            singe_kph = find_eigen_kpar(phcs, k0s, qa, num.modes, mode="H")
            kpe.append(singe_kpe)
            kph.append(singe_kph)
            
            if nik0%flagenum == 0:
                iky = int(nik0/Nk0*50)
                aii = "*" * iky
                bii = "." * (50 - iky)
                cii = iky / 50 * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                      end = "")  
            
        print("\n" + "Initialization accomplished.")
        
            
        maxky = np.sqrt(maxk0**2 - qa**2)

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
        print("=================")
        start = time.time()
        print("Meshing:")
        kylist = np.arange(0, maxky, delta)
        
        flagenum = len(kylist)//50
        niky = 0
        for ky in kylist:
            niky = niky + 1
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
                        #if kpei[0]<ky and kphi[0]<ky:
                        lenthe = len(kpei)
                        lenthh = len(kphi)
                        if lenthe>2 and lenthh>2:
                            imkpe = kpe[i][1]*1
                            imkph = kph[i][1]*1
                            imkpe.extend(kpei[0:lenthe-2])
                            imkph.extend(kphi[0:lenthh-2])
                            imkpe = imkpe[0:num.imag]
                            imkph = imkph[0:num.imag]
                            
                            kppe = [kpei[lenthe-2:], imkpe]
                            kpph = [kphi[lenthh-2:], imkph]
                            
                            k_para_e.append(kppe)
                            k_para_h.append(kpph)
                        else:
                            k_para_e.append(kpe[i])
                            k_para_h.append(kph[i])
            
            if niky%flagenum == 0:
                iky = int(niky/len(kylist)*50)+1
                aii = "*" * iky
                bii = "." * (50 - iky)
                cii = iky / 50 * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                      end = "")  
            
        print("\n" + "Mesh accomplished.")
        
        
        self.kyk0 = kyk0
        self.k_para_e = k_para_e
        self.k_para_h = k_para_h
        
    
    def run(self, limit=0.999):
        """get the BICs.
        
        Parameters
        ----------
        limit:float
            the precision of judging if a point in q-k0 space is a BIC
        
        """
        kyk0 = self.kyk0
        k_para_e = self.k_para_e
        k_para_h = self.k_para_h
        phcs = self.phcs
        num = self.num
        n_real = num.real
        h = phcs.h
        even_coefs = []
        odd_coefs = []
        dataky = []
        datak01 = []
        datak02 = []
        print("=================")
        print("Searching:")
        start = time.time()
        flagenum = len(kyk0)//50
        
        ky = kyk0[0][0]
        k0 = kyk0[0][1]
        for i in range(len(kyk0)):
            ky = kyk0[i][0]
            k0 = kyk0[i][1]
            if i:
                if kyk0[i][0] == kyk0[i-1][0]:
                    pass
                else:
                    datak01.append(k0/(2*np.pi))
                    datak02.append(kyk0[i-1][1]/(2*np.pi))
                    dataky.append(ky/(2*np.pi))
            else:
                dataky.append(ky/(2*np.pi))
                datak01.append(k0/(2*np.pi))
            
            kparae = k_para_e[i]
            kparah = k_para_h[i]
            f1 = FieldsWithCTIRMix(phcs, num, k0, 0, ky, kparae, kparah)
            even_coefs_inside = f1.even_coefs_inside
            odd_coefs_inside = f1.odd_coefs_inside
            even_coefs.append(even_coefs_inside)
            odd_coefs.append(odd_coefs_inside)
            
            if i%flagenum == 0:
                iky = int(i/len(kyk0)*50) + 1
                aii = "*" * iky
                bii = "." * (50 - iky)
                cii = iky / 50 * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(cii,aii,bii,dur),
                      end = "")      
        datak02.append(k0/(2*np.pi))
        print("\n"+"Search accomplished.")
        
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
            odd_coefs_boundry = np.real(odd_coefs).tolist()
            even_coefs_boundry = np.real(even_coefs).tolist()

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
            limitdelta = 0.02 * (2*np.pi)
            
            for i in range(len(test) - 1):
                for j in range(len(bicregion)):
                    if (abs(bicregion[j][-1][0] - test[i+1][0])<limitdelta
                        and 
                        abs(bicregion[j][-1][1] - test[i+1][1])<limitdelta):
                        bicregion[j].append(test[i+1])
                        flag = 0
                        break
                    flag = 1
               
                if flag:
                    bicregion.append([test[i+1]])
            
            bic_ky = []
            bic_k0 = []
            for onebicregion in bicregion:
                onebic = [onebicregion[0][0],
                          onebicregion[0][1],
                          onebicregion[0][2]]
                for j in range(len(onebicregion)-1):
                    if onebicregion[j+1][-1]>onebic[-1]:
                        onebic = [onebicregion[j+1][0],
                                  onebicregion[j+1][1],
                                  onebicregion[j+1][2]]
                bic_ky.append(onebic[0]/(2*np.pi))
                bic_k0.append(onebic[1]/(2*np.pi))
            return bic_ky, bic_k0
        try:
            bic_ky, bic_k0 = find_bic(h)
        except:
            bic_ky, bic_k0 = [], []
        if bic_ky:
            self.bic_kys = bic_ky
            self.bic_k0s = bic_k0
        else:
            self.bic_kys = []
            self.bic_k0s = []
        print("Number of BICs found: ", len(bic_k0))
        
        dataky = np.array(dataky)
        datak01 = np.array(datak01)
        datak02 = np.array(datak02)
        self.dataky = dataky
        self.datak01 = datak01
        self.datak02 = datak02
        
        self.showbic()
    
    def showbic(self):
        """show bics in the k-omega space.
        """
        phcs = self.phcs
        qa = self.qa
        h = phcs.h
        bic_ky = self.bic_kys
        bic_k0 = self.bic_k0s
        dataky = self.dataky
        datak01 = self.datak01
        datak02 = self.datak02
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.set_xlabel('$k_y(2\pi/a)$')
        ax.set_ylabel('$\omega(2\pi c/a)$')
        ax.set_title("BICs in $k_y-\omega$ space($h="+\
                                              str(round(h,3))+\
                                              "a, q="+str(qa)+"$)")
        ax.plot(dataky, datak01, 'b', ls=':')
        ax.plot(dataky, datak02, 'black', ls='--')
        ax.fill_between(dataky, datak01, datak02,
                        color='C1', alpha=0.3,
                        interpolate=True,
                        label="Searching range")
        ax.scatter(bic_ky, bic_k0, marker='*',
                   s=100, c="red", edgecolors="black", 
                   label="BIC")
        plt.legend(markerscale=1)
        plt.show()
        
        
        
     
        
