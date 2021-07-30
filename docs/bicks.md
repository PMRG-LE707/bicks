# bicks package

## Submodules

## bicks.bicsearch module


### class bicks.bicsearch.FindBICs(phcs, num, mode='E', Nq=250)
Bases: `object`

find BICs in q-k0 space with single polarization.


#### BIC_qs()
each item contains BICs’ qs for corresponding thickness.


* **Type**

    list[list[float]]



#### BIC_k0s()
each item contains BICs’ k0s for corresponding thickness.


* **Type**

    list[list[float]]



#### BIC_hs()
the thickness of PhC slab where BICs exist.


* **Type**

    list[float]



#### dynamicplot(save=False)

#### find_kpar(mode, Nq)

#### getcoeffs()
get the ratio of coefficients of two Bloch waves in opposite
direction.


#### run(hstart, hend, Nh=20, limit=0.999)
search BICs by varying thickness of PhC slab.


* **Parameters**

    
    * **hstart** (*float*) – start searching in this thickness


    * **hend** (*float*) – end searching in this thickness


    * **Nh** (*int**, **optional*) – number of searching thickness


    * **limit** (*float*) – the precision of judging if a point in q-k0 space is a BIC



#### showbic(i=0)

### class bicks.bicsearch.FindBICsMix(phcs, num, qa, k0range=3.141592653589793, Nk0=200)
Bases: `object`

find BICs in ky-k0 space with mix polarization.


#### BIC_kys()
each item contains BICs’ qs for corresponding thickness.


* **Type**

    list[float]



#### BIC_k0s()
each item contains BICs’ k0s for corresponding thickness.


* **Type**

    list[float]



#### run(limit=0.999)
get the BICs.


* **Parameters**

    **limit** (*float*) – the precision of judging if a point in q-k0 space is a BIC



#### showbic()
## bicks.boundryconditionwithcTIR module


### bicks.boundryconditionwithcTIR.getcoemix(real_fields, imag_fields, num, constant_number=2)
For the mix mode(both E and H mode)


* **Parameters**

    
    * **real_fields** (*list**[**FieldInPhCS**]*) – the fields from eigenstates with real kz


    * **imag_fields** (*list**[**FieldInPhCS**]*) – the fields from eigenstates with imaginary kz


    * **num** (*EssentialNumber*) – 


    * **constant_number** (*int**, **optional*) – the serial number of columm of extend matricx which represents
    constant in eqs.



* **Returns**

    
    * *list[float]* – the ratio of coefficients of two Bloch waves in opposite
    direction(the tangential compoments of E are even in z direction).


    * *list[float]* – the ratio of coefficients of two Bloch waves in opposite
    direction(the tangential compoments of E are odd in z direction).


    * *list[float]* – real kz of all Bloch waves




### bicks.boundryconditionwithcTIR.getcoesingle(real_fields, imag_fields, num, constant_number=0)
For the single mode(only E or H mode)


* **Parameters**

    
    * **real_fields** (*list**[**FieldInPhCS**]*) – the fields from eigenstates with real kz


    * **imag_fields** (*list**[**FieldInPhCS**]*) – the fields from eigenstates with imaginary kz


    * **num** (*EssentialNumber*) – 


    * **constant_number** (*int**, **optional*) – the serial number of columm of extend matricx which represents
    constant in eqs.



* **Returns**

    
    * *list[float]* – ratio of coefficients of two Bloch waves in opposite
    direction(the tangential compoments of E are even in z direction).


    * *list[float]* – ratio of coefficients of two Bloch waves in opposite
    direction(the tangential compoments of E are odd in z direction).


    * *list[float]* – real kz of all Bloch waves




### bicks.boundryconditionwithcTIR.singleboundry(real_fields, imag_fields, num, constant_number=0)
For the mix mode(both E and H mode)


* **Fields**

    a list of lenth 4, it contains incident and
    reflected fields with real kz and imag kz, respectively.



* **Nne**

    negative diffraction oders



* **Npo**

    positive diffraction oders



* **Constant_number**

    the serial number of columm which is
    constant in eqs.



* **Returns**

    the coefficents of different eigenstates in two
    kinds(even or odd for E mode)


## bicks.crystalandnumber module


### class bicks.crystalandnumber.EssentialNumber(n_radiation=1, nimag_plus=0, n_propagation=0)
Bases: `object`

Some essential number of modes or orders.


#### ne()
number of diffraction orders(negetive).


* **Type**

    int(>0)



#### po()
number of diffraction orders(positive).


* **Type**

    int(>0)



#### d()
number of diffraction orders.


* **Type**

    int(>0)



#### r()
number of radiation channels in air.


* **Type**

    int(>0)



#### listr()
radiation channels orders.


* **Type**

    np.ndarray(dtype=np.int)



#### real()
number of considered real kz.


* **Type**

    int(>0)



#### imag()
number of considered imag kz.


* **Type**

    int(>=0)



#### modes()
number of considered kz; modes = real + imag.


* **Type**

    int(>=0)



### class bicks.crystalandnumber.PhotonicCrystalSlab(epsilon, fillingrate, mu=array([1, 1]), thickness=1.0, periodlength=1)
Bases: `object`

Here, we defined a class named PhotonicCrystalSlab
which is 1D.
Warning! The structure is non-magnetic.


#### h()
thickness of the PC slab.


* **Type**

    float



#### ep()
a list which contains the dielectric constant
of the two different layers; [small, big],
for example, [1.0, 4.9]


* **Type**

    list



#### fr()
filling ratio (fill the small dielectric constant medium).


* **Type**

    float



#### a()
the length of a period.


* **Type**

    float



#### show()
Show the PhC slab in a picture.

## bicks.eigenkpar module


### bicks.eigenkpar.find_eigen_kpar(phcs, k0a, qa, nmode, mode='E')
The eigenstates of the 1D photonic crystal.

phcs: PhotonicCrystalSlab

    the Photonic Crystal Slab which is a kind of class.

qa: float

    the Bloch wave number

k0a: float

    the frequency divided by (2pi\*c)

nmode: int

    the number of considered Bloch modes

mode: {“E”, “H”}, optional

    the mode of the eigenstate


* **Returns**

    
    * *np.ndarray* – real k_parallels(eigenvalue) of eigenstates


    * *np.ndarray* – imaginary k_parallels(eigenvalue) of eigenstates




### bicks.eigenkpar.find_eigen_kpar_in_an_area(phcs, qa, k0a, num, kpara_real_extreme, kpara_imag_extreme, mode='E')
The eigenstates of the 1D photonic crystal.

phcs: PhotonicCrystalSlab

    the Photonic Crystal Slab which is a kind of class.

qa: float

    the Bloch wave number

k0a: float

    the frequency divided by (2pi\*c)

num: EssentialNumber
kpara_real_extreme: list[float]

> there is a real eigenvalue between any the adjacent two in this list

kpara_imag_extreme: list[float]

    there is an imaginary eigenvalue between any the adjacent two in this
    list

mode: {“E”, “H”}, optional

    the mode of the eigenstate


* **Returns**

    
    * *np.ndarray* – real k_parallels(eigenvalue) of eigenstates


    * *np.ndarray* – imaginary k_parallels(eigenvalue) of eigenstates



## bicks.field module


### class bicks.field.BulkEigenStates(phcs, k0a, kpar, qa, mode='E', normalization=1)
Bases: `object`

The eigenstates i.e. periodic parts of fields in PhC


#### Fourier_coefficients(i)

#### u(xra)

#### v(xra)

#### w(xra)

### class bicks.field.FieldInPhcS(eigenstate, kya=0)
Bases: `object`

The field generated by a BulkEigenStates in PhC.


#### Ex(x, z, kzdirection=0)

#### Ey(x, z, kzdirection=0)

#### Hx(x, z, kzdirection=0)

#### Hy(x, z, kzdirection=0)

#### fieldfc(i, field_compoments)
without the basis


#### show(fieldcomponent, oprator, Nx=20)
unit in period a


### class bicks.field.FieldsWithCTIRInArea(phcs, num, k0a, qa, kya, real_parallel, imag_parallel, mode='E')
Bases: `object`

The class can get the coefficinents of different fields
when the cTIR happends on the upper boundry in specific
q-k0 area dicided by EssentialNumber num.


### class bicks.field.FieldsWithCTIRMix(phcs, num, k0a, qa, kya, kpe, kph)
Bases: `object`

## bicks.mathtool module


### bicks.mathtool.dichotomy(f, a, b, epsilon=1e-05)
Tradional dichotomy to find a root of a function


### bicks.mathtool.find_all_peaks(f, x_start, x_end, deltax=0.01, eps=0.001, lastdata=[])

### bicks.mathtool.find_n_roots(f, n, deltax, eps=1e-10)
Warning! Don’t make the deltax = 0.1


### bicks.mathtool.find_n_roots_for_small_and_big_q(f, qa, n, gox=0, deltax=0.024, eps=1e-10, peak1=0)

### bicks.mathtool.find_proj_roots(f, endk0, startk0=0.121, deltak0=0.12, eps=1e-10)

### bicks.mathtool.find_real_roots(f, endkz, startkz=0, deltakz=0.12, eps=1e-10)

### bicks.mathtool.find_real_roots_for_small_and_big_q(f, qa, deltax=0.024, eps=1e-10)

### bicks.mathtool.golden_section(f, a, b, epsilon=1e-10)

### bicks.mathtool.minus_cosqa(x)

### bicks.mathtool.secant(f, a, b, eps=1e-05)
## bicks.photoniccrystalbandprojection module


### bicks.photoniccrystalbandprojection.find_band_projection(phcs, num, Nq=100, mode='E')
find the area where we can use dichotomy to find roots(kz).

phcs: PhotonicCrystalSlab

    the Photonic Crystal Slab which is a kind of class.

num: EssentialNumber
mode: {“E”, “H”}, optional

> the mode of the eigenstate

Nq: int, optional

    number which we divided half of Brillouin into


* **Returns**

    
    * **k0_floor** (*np.ndarray*) – real k_parallels of eigenstates


    * **imag_k_parallel** (*np.ndarray*) – imag k_parallels of eigenstates




### bicks.photoniccrystalbandprojection.mini_frequncy(phcs, num, qa, deltak0)
Find the floor of frequncy range for a specific q where the number of real
k is a constant which is a paramter.

phcs: PhotonicCrystalSlab
num: EssentialNumber


* **Returns**

    **k0_floor** – the floor of range of k0 where the number of real k is a constant.



* **Return type**

    np.ndarray


## Module contents
