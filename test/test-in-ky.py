# -*- coding: utf-8 -*-
from bicks.crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicks.bicsearch import FindBICsMix
import numpy as np

fr = 0.5 # 填充率
ep = [1.0, 5.5] # 两种介电常数
phcs = PhotonicCrystalSlab(ep, fr, thickness=1.2) # 生成光子晶体平板，默认的厚度和周期都为1，调整周期的大小没有意义，因为之后的长度单位都是以周期为单位的
n_radiation_channel = 1 # 辐射通道的个数
num = EssentialNumber(n_radiation=n_radiation_channel, nimag_plus=2) # 创建一个“必要数字”的实例，这是划分区域的关键
fb = FindBICsMix(phcs, num, 0, k0range=0.3*2*np.pi) # 由光子晶体、划分的区域以及偏振模式构建的BIC搜寻器
fb.run() # 将反射相位记录到内存中