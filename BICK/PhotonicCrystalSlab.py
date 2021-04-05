import matplotlib.pyplot as plt


class PhotonicCrystalSlab:
    """
    Here, we defined a class named PhotonicCrystalSlab
    which is 1D.
    Warning! The structure is non-magnetic.

    Attributes:
    ------------
    h: thickness of the PC slab.
    ep: a list which contains the dielectric constant
    of the two different layers, most of time, the first is
    the air.
    fr: filling ratio (fill the 'air').
    a: the length of a period.
    """
    def __init__(self, thickness, epsilon, fillingrate, periodlength, mu=[1, 1]):
        self.h = thickness
        self.ep = epsilon
        self.fr = fillingrate
        self.a = periodlength
        self.mu = mu
    
    def show(self):
        """
        Show the PhC slab
        """
        # two rectangles' paramters
        hight = self.h / self.a
        width1 = (1 - self.fr)
        width2 = self.fr
        
        # set up the canvas
        font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size'   : 14}
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

    Attributes:
    ------------------
    ne: number of diffraction orders(negetive)
    po: number of diffraction orders(positive)
    d: number of diffraction orders
    r: number of radiation channels in air
    list_r: radiation channels orders
    overlap: number of equal ai and ri
    real: number of real kz for one q and omega
    imag: number of considered imag kz for one q and omega 
    """
    def __init__(self, n_ne, n_po, n_propagation, n_radiation):
        """
        n_ne: number of diffraction orders(negetive)
        n_po: number of diffraction orders(negetive)
        n_radiation: number of radiation channels in air
        n_propagation: number of real kz for one q and omega
        """
        
        n_d = n_ne + n_po + 1
        if n_radiation%2:
            temlistr = [i - (n_radiation - 1)/2 for i in range(n_radiation)]
        else:
            temlistr = [i - (n_radiation - 2)/2 for i in range(n_radiation)]
        list_r = [int(i + n_ne) for i in temlistr]
        n_e_imag = n_d + n_radiation - 2 * n_propagation + 1
        n_e_overlap = n_e_imag
        n_m_imag = n_e_imag - 1
        n_m_overlap = n_m_imag
        
        self.ne = n_ne
        self.po = n_po
        self.d = n_d
        self.r = n_radiation
        self.list_r = list_r
        self.overlap_e = n_e_overlap
        self.overlap_m = n_m_overlap
        self.real = n_propagation
        self.imag_e = n_e_imag
        self.imag_m = n_m_imag
