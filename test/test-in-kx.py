# -*- coding: utf-8 -*-
from bicks.crystalandnumber import PhotonicCrystalSlab, EssentialNumber
from bicks.bicsearch import FindBICs

fr = 0.2 # 填充率
ep = [1.0, 4.5] # 两种介电常数
phcs = PhotonicCrystalSlab(ep, fr) # 生成光子晶体平板，默认的厚度和周期都为1，调整周期的大小没有意义，因为之后的长度单位都是以周期为单位的
n_radiation_channel = 2 # 辐射通道的个数
num = EssentialNumber(n_radiation=n_radiation_channel) # 创建一个“必要数字”的实例，这是划分区域的关键
mode = "E" # 设置为TE偏振，如果要设置为TM偏振请键入"H"
fb = FindBICs(phcs, num, mode=mode) # 由光子晶体、划分的区域以及偏振模式构建的BIC搜寻器

fb.getcoeffs() # 将反射相位记录到内存中
hstart = 1.5 # 厚度的下限
hend = 3 # 厚度的上限
Nh = 50 # 遍历厚度的个数
fb.run(hstart, hend, Nh=Nh) # 让我们开始寻找吧!
