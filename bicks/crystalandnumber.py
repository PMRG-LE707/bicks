import matplotlib.pyplot as plt
import numpy as np


class PhotonicCrystalSlab:
    """Here, we defined a class named PhotonicCrystalSlab
    which is 1D.
    Warning! The structure is non-magnetic.

    Attributes
    ----------
    h: float
        thickness of the PC slab.
    ep: list
        a list which contains the dielectric constant
        of the two different layers; [small, big],
        for example, [1.0, 4.9]
    fr: float
        filling ratio (fill the small dielectric constant medium).
    a: float
        the length of a period.
    """
    def __init__(self, epsilon,
                 fillingrate,
                 mu=np.array([1, 1]),
                 thickness=1.0,
                 periodlength=1):
        """
        Initialize the 1D PhC slab.
        
        Paramters
        ---------
        epsilon: list[float]
            a list which contains the dielectric constant
            of the two different layers   
        fillingrate: float
            filling ratio (fill the small dielectric constant medium).
        periodlength: float
            the length of a period.

        """
        self.h = thickness
        self.ep = np.array(epsilon, dtype=np.float32)
        self.fr = fillingrate
        self.a = periodlength
        self.mu = mu
    
    def show(self):
        """
        Show the PhC slab in a picture.
        """
        # two rectangles' paramters
        hight = self.h / self.a
        width1 = (1 - self.fr)
        width2 = self.fr
        
        # set up the canvas
        font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size': 14}
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.set_xlabel('x (a)', font)
        ax1.set_ylabel('z (a)', font)
        ax1.set_title("PhC Slab(one period)", font)
        ax1.set_xlim([-0.5, 1.5])
        ax1.set_ylim([-hight * 0.6, hight])
        
        # first rectangle
        ax1.add_patch(
            plt.Rectangle(
                (0, - hight / 2),  # (x,y)
                width1,  # width
                hight,  # height
                color='maroon', 
                alpha=0.5
            )
        )
        # second rectangle
        ax1.add_patch(
            plt.Rectangle(
                (width1, - hight / 2),  # (x,y)
                width2,  # width
                hight,  # height
                color='blue', 
                alpha=0.5
            )
        )
        
        # plot the arrow and delectric constants
        txtname = ['$\epsilon_{1} = ' + str(self.ep[0]) + '$',
                   '$\epsilon_{2} = ' + str(self.ep[1]) + '$']
        ax1.annotate(txtname[0],
            xy=(width1/2, hight / 2 * 0.9), xycoords='data',
            xytext=(0.1, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round4", fc="maroon", alpha=0.3),
            arrowprops=dict(arrowstyle="fancy",
                                  connectionstyle="arc3,rad=0.2",
                                  fc="maroon", alpha=0.3))
        ax1.annotate(txtname[1],
            xy=(width1 + width2 / 2, hight / 2 * 0.9), xycoords='data',
            xytext=(0.7, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round4",  fc="blue", alpha=0.3),
            arrowprops=dict(arrowstyle="fancy",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="blue", alpha=0.3))
        plt.show()
        
class EssentialNumber:
    """
    Some essential number of modes or orders.

    Attributes
    ----------
    ne: int(>0)
        number of diffraction orders(negetive).
    po: int(>0)
        number of diffraction orders(positive).
    d: int(>0)
        number of diffraction orders.
    r: int(>0)
        number of radiation channels in air.
    listr: np.ndarray(dtype=np.int)
        radiation channels orders.
    real: int(>0)
        number of considered real kz.
    imag: int(>=0)
        number of considered imag kz.
    modes: int(>=0)
        number of considered kz; modes = real + imag.
    """
    def __init__(self, n_radiation=1, nimag_plus=0,
                 n_propagation=0):
        """
        Initialize the essential numbers
        
        Paramters
        ---------
        n_radiation: int, optional
            number of radiation channels in air.
        nimag_plus: int, optional
            considered more imag modes,
            the more orders considered the better accurcy got.
        n_propagation: int, optional
            number of real kz for one Bloch q and frequency,
            the number is 1 more than n_radiation or 3 while 
            n_radiation is 1.
        """

        self.r = n_radiation
        
        if n_propagation==0:
            n_propagation = n_radiation + 1
        
        self.real = n_propagation
        self.imag = 1 + nimag_plus
        self.modes = self.real + self.imag
        self.d = self.modes
        if (self.d)%2:
            self.ne = self.d // 2 
            self.po = self.d // 2 
        else:
            self.ne = self.d // 2
            self.po = self.d // 2 - 1
            
        if n_radiation%2:
            listr = [i - (n_radiation - 1) // 2
                     for i in range(n_radiation)]
        else:
            listr = [i - n_radiation // 2
                     for i in range(n_radiation)]
    
        if n_propagation-n_radiation==1:
            
            self.listr = np.array(listr, dtype=np.int) 
                
        elif n_propagation == 3 and n_radiation == 1:
            listr.append(-1)
            self.listr = np.array(listr, dtype=np.int)
              
        else:
            raise ValueError("""n_propagation should be 1 more 
                             than n_radiationor or 3 while n_radiation == 1""")
            