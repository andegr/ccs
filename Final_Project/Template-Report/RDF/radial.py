#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:54:38 2023

@author: sebastiengroh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rc


########## translation ##########

rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'font.size': 22})
plt.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 24

fig , ax = plt.subplots(figsize=(8.27,11.69/2))

plt.gcf().subplots_adjust(bottom=0.15) # leave room at the bottom for axe's label
plt.gcf().subplots_adjust(left=0.17)

filein = open("radial.txt","r")
r = []
g = []
g1 = []
for i in range(100):
    line=filein.readline()
    part = line.split()
    r.append(float(part[0])/2.55e-10)
    g.append(float(part[1]))
    g1.append(1)
    
filein.close()
ax.set_xlabel(r"r [$\sigma]$", font="Times", fontsize=32 )
ax.set_ylabel("g(r)", font="Times", fontsize=32 )
plt.plot(r, g)
plt.plot(r, g1, linestyle="dashed")
plt.xlim(0,4)
plt.ylim(0,2)
plt.show()
fig.savefig("rdf.pdf")