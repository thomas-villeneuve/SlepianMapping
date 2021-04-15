# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:51:12 2020

@author: Duncan
"""

##############################################################################
'''
                PACKAGE IMPORTATION
'''
##############################################################################


import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt


##############################################################################
'''
                SYSTEM PARAMETERS
'''
##############################################################################


Time = 1000; #Must be decently large. Can show ideal value by convergence plot
tp = int (Time/2); #The expression we had was only valid for long times of v(t)
N= 100; #One should verify by a convergence plot that this is sufficient

##############################################################################
'''
                HERE I HAVE FITTING FUNCTIONS
'''
##############################################################################

def func(x, a, b):
    
    return a * np.exp(-b * x)

##############################################################################
'''
                SIMULATION FUNCTIONS
'''
##############################################################################

def bestfitparameters(Time, N, tp, alp):
    
    #This performs a best fit to <v(t)v(t')>
    
    v = np.zeros([Time, N]);
    
    for t in range(1, Time):
    
        s = (2*np.random.randint(0, 2, size=N)) - 1;
        
        v[t] = (1-alp)*v[t-1] + s;
    
    
    prod = np.zeros([Time, N]);
    ave = np.zeros([Time-tp, N]);
    
    for n in range(0, N):
        
        g = v[tp:, n];
    
        mat = np.outer(g,g);
        
        for t in range(0, Time-tp):
            ave[t, n] = np.mean(np.diagonal(mat, t));
            
    T=range(0, Time-tp);
    val=np.zeros(Time-tp); #Initializes 
    
    for t in range(0, Time-tp):
        val[t] = np.mean(ave[t, :]);
        
        
    popt, pcov = sc.curve_fit(func, T, val);
    
    
    #I have included ploting if you want to check the exponential dependence.
    
    #plt.figure('Ave')   
    #plt.plot(T, func(T, popt[0],popt[1]))
    #plt.plot(T, val)
    #plt.xlabel('t-t_')
    #plt.ylabel('<x(t) x(t_)>')
    #plt.legend(['Numerical', 'Fit'])
    #plt.show()
    print('Check')
    
    return popt, pcov 
    
def alpdependence(Time, N, tp, alpmin, alpmax, number):
    
    #This spits out best fit parameters as we change alpha
    
    alph = np.linspace(alpmin, alpmax, number);
    alphsize = np.size(alph);
    A = np.zeros(alphsize)
    B = np.zeros(alphsize)
    Aerr = np.zeros(alphsize)
    Berr = np.zeros(alphsize)
    
    for i in range(0, alphsize):
        P = bestfitparameters(Time, N, tp, alph[i]);
        A[i] = P[0][0]; B[i] = P[0][1];
        Aerr[i] = np.sqrt(P[1][0][0]); Berr[i] = np.sqrt(P[1][1][1]);
        
    return A, B, alph, Aerr, Berr

##############################################################################
'''
        NOW WE RUN THE CODE
'''
##############################################################################

M = alpdependence(Time, N, tp, 0.001, 0.01, 20);

plt.figure('A vs alpha')
plt.plot(M[2], M[0], '-o')
plt.errorbar(M[2], M[0], yerr=M[3], fmt='-')
plt.xlabel('alpha')
plt.ylabel('A')

plt.figure('B vs alpha')
plt.plot(M[2], M[1], '-o')
plt.errorbar(M[2], M[1], yerr=M[4], fmt='-')
plt.xlabel('alpha')
plt.ylabel('B')

plt.show()

'''
This code is by no means fully optimized. If you have free time, 
consider trying to remove for loops in the above
'''

'''
Here I've included a little sample code of pythons efficiency at array based operations:

import time    
    
Lx=1000
Ly=1000

A = np.random.rand(Lx,Ly)
B = np.random.rand(Lx,Ly)
C = np.random.rand(Lx, Ly)

t = time.process_time()

for i in range(0, Lx):
    for j in range(0, Ly):
        C[i, j] = C[i, j]+(B[i, j]*A[i, j]);
        
print(time.process_time()-t)


tt = time.process_time()
#print(tt)

for i in range(0, Lx):
    C[i, :] = C[i, :]+(B[i, :]*A[i, :]);
    
ttt= time.process_time()              
print(ttt-tt)

tttt = time.process_time()
#print(tt)

C = C + (B*A);
        
ttttt= time.process_time()              
print(ttttt-tttt)

'''
