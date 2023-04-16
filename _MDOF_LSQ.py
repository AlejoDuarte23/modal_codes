import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys
import warnings
import scipy.stats as stats
from scipy import signal


# Theor PSD 
def H(freq,f,z,S,Se,Nm,N):
    H = np.zeros((Nm,Nm,N),dtype=np.complex_)
    for i in range(0,Nm):
        for j in range(0,Nm): 
            if i==j:
                bki = f[i]/freq
                bkj = f[j]/freq
                ter1 = 1/((1-bki)+2j*z[i]*bki)
                ter2 = 1/((1-bkj)+2j*z[j]*bkj)
                H[i,j,:] = (10.**S[i,j])*(ter1)*(ter2)+10.**Se
    return H
#--------------------------- 1. Likely ---------------------------------------#
def likelihood(x,freq,ymed,Nm,N):
  
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]
    modelo = np.array([])
    H1 = H(freq,f,z,S,Se,Nm,N)
    for i in range(len(freq)):
     
        ESY = H1[:,:,i]+10**-40
        modelo = np.abs(np.append(modelo,np.trace(ESY)/N))
    return modelo-ymed

#--------------------------- 2. Plot PSD --------------------------------------#
def plot_psd(x,Nm,N,freq_id,s1_id):
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]
    H1 = H(freq_id,f,z,S,Se,Nm,N)
    plt.figure(1)
    for i in range(Nm):
        plt.plot(freq_id,10*np.log10(np.trace(H1)/N),'r',label = 'E[Sy]')
    plt.plot(freq_id,10*np.log10(s1_id),'b',label ='Single Value Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Decibel [Db]')
#--------------------------- 5. fdd ------------------------------------------#
def fdd(Acc,fs,Nc):
    # Acc: Acceleration Matriz NcxN
    # fs:  Sampling Frequency
    # Nc:  Number of channels
    AN = nextpow2(Acc)
    # Memory alocation for the matrix
    PSD = np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex_)
    freq= np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex_)

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(Acc[:,i], Acc[:,j], fs, nfft=AN,nperseg=2**11,noverlap = None,window='hamming')
            freq[i,j]= f
            PSD[i,j]= Pxy
           
    #eigen values descomposition 
    s1 = np.zeros(len(f))
    for  i in range(len(f)):
        u, s, vh = np.linalg.svd(PSD[:,:,i], full_matrices=True)
        s1[i] = s[0]
    return s1,PSD,f

#--------------------------- 5. Load txt -------------------------------------#
def MDOF_LSQ(xo,ACC,fs,fo,fi,Nm):
    s1,psd,freq = fdd(ACC,fs,len(ACC[1,:]))
    idd = (np.where((freq>= fo) & (freq <= fi)))
    freq_id= freq[idd]
    s1_id= s1[idd]
    N = len(freq_id)
    # Single valur spectrum
    plt.figure()
    plt.plot(freq_id,10*np.log10(s1_id))
    likelyhood = lambda xo,freq,si: likelihood(xo,freq,si,Nm,N)
    opt = least_squares(likelyhood ,xo,loss='cauchy',f_scale=0.1,args=(freq_id, s1_id))
    plot_psd(opt.x,Nm,N,freq_id,s1_id)
    return opt,psd

def nextpow2(Acc):
    N = Acc.shape[0]
    _ex = np.round(np.log2(N),0)
    Nfft = 2**(_ex+1)
    return int(Nfft)

def CPSD(Acc,fs,Nc,fo,fi):
    # Acc: Acceleration Matriz NcxN
    # fs:  Sampling Frequency
    # Nc:  Number of channels
    AN = nextpow2(Acc)
    # Memory alocation for the matrix
    PSD = np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex_)
    freq= np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex_)

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(Acc[:,i], Acc[:,j], fs, nfft=AN,nperseg=2**11,noverlap = None,window='hamming')
            freq[i,j]= f
            PSD[i,j]= Pxy
    TSx = np.trace(PSD)/len(f)      
    idd = (np.where((f>= fo) & (f <= fi)))
    freq_id= f[idd]
    TSxx= np.abs(TSx[idd])
    N = len(freq_id)
    
    return freq_id,TSxx,N,len(f)

def Model(x,freq,Nm,N,Fc):
    # breakpoint()
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]
    modelo = np.array([])
    H1 = H(freq,f,z,S,Se,Nm,N)
    for i in range(len(freq)):
     
        ESY = H1[:,:,i]+10**-40
        modelo = np.abs(np.append(modelo,np.trace(ESY)/Fc))
    return modelo

