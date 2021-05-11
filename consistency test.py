#!/usr/bin/env python
# coding: utf-8

# In[43]:


from starry.kepler import Primary, Secondary, System
import numpy as np
import slepian
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from IPython.display import display, Math
np.random.seed(264)


# In[44]:


def changeOfBasis(L):
    """Change of basis matrix."""
    return np.array(slepian.slepianToHarmonic(L),dtype=float)


# In[45]:


def setCoeffs(X,r,planet):
    """Takes a harmonic vector X and sets the starry map parameters."""
    
    # Pull out the Y0,0 coefficient X0 to set L
    X0 = X[0] 
    
    # Renormalize so that X0 = 1
    X = X/X0 
    
    # Set luminosity
    planet.L = r*X0 
    
    # Set map coefficients
    planet[1,:] = X[1:] 
    
    return


# In[46]:


def getLightcurve(X,params,time,star,planet,L):
    """Creates a planet-star system and generate the planet lightcurve."""
    
    # Extract parameters (more to be added later)
    r = params
    
    # Set map coefficients
    setCoeffs(X,r,planet)
    
    # Instantiate the system
    system = System(star, planet)
    
    # Compute the flux
    system.compute(time)
    flux = planet.lightcurve
    
    return flux


# In[47]:


def lnPrior(X):
    """Log prior probability."""
    
    # Map coefficents should be |X| < 1 and 
    # X0 > 0 so that luminosity is positive. 
    if np.any(X < -1) or np.any(X > 1) or (X[0] <= 0): 
        return -np.inf
    else:
        return 0


# In[48]:


def lnLike(T,params,time,flux,flux_err,star,planet,L,C):
    """Log likelihood."""
    
    # Convert from slepian to harmonic vector.
    X = np.dot(C,T) 
    
    # Check that X fits the priors.
    l = lnPrior(X)
    if np.isinf(l):
        return l
    
    # Create a system and get the computed flux.
    c_flux = getLightcurve(X,params,time,star,planet,L)
    
    # Caluculate log likelihood.
    like = np.sum(((flux - c_flux)/flux_err)**2 + np.log(2*np.pi*flux_err**2))
    l += -0.5*like
    
    return l


# In[49]:


def maxLike(lnlike,guess,params,time,flux,flux_err,star,planet,L,C):
    """Maximum likelihood solution for slepian coefficients."""
    
    # Define negative log likelihood
    nlnLike = lambda *args: -lnLike(*args)

    # Find the numerical optimum of the likelihood function
    soln = minimize(nlnLike, guess, args = (params,time,flux,flux_err,star,planet,L,C),method='Nelder-Mead')
    T_maxl = soln.x

    return T_maxl


# In[50]:


#Instantiate a star
star = Primary()

# Instantiate the planet
# We'll use the same planet in MCMC but change the params each time as necessary 
planet = Secondary(L)
planet.a = 5 # Orbital semi-major axis in units of the stellar radii.
planet.inc = 87 # Orbital inclination (90 degrees is an edge-on orbit)
planet.porb = 1 # Orbital period in Julian days
planet.prot = 1 # Rotation period in days (synchronous if prot=porb)
#planet.ecc = 0 # Eccentricity
#planet.w = 0  # Longitude of pericenter in degrees (meaningless for ecc=0)
planet.tref = 0.5 # Time of transit in Julian days

# Generate the lightcurve for true planet 
S1 = 0.7
S2 = 0.3
S3 = 0.0
S4 = 0.0

T_true = np.array([S1,S2,S3,S4])
X_true = np.dot(C,T_true)

params = 0.01

flux_true = getLightcurve(X_true,params,time,star,planet,L)

# Add noise
flux_err = 0.001
flux = flux_true + flux_err*np.random.randn(1000)


# In[51]:


# Define a guess
guess = np.array([0.1,0.1,0.1,0.1])

# Find the max likelihood solution
T_maxl = max_likelihood(lnlike,guess,params,time,flux,flux_err,star,planet,L,C)

# Print results
print("Maximum likelihood estimates:")
print("T_1 = {0:.3f}".format(T_maxl[0]))
print("T_2 = {0:.3f}".format(T_maxl[1]))
print("T_3 = {0:.3f}".format(T_maxl[2]))
print("T_4 = {0:.3f}".format(T_maxl[3]))


# Plot max likihood flux
X_maxl = np.dot(C,T_maxl)
flux_maxl = getLightcurve(X_maxl,params,time,star,planet,L)

plt.figure(figsize=(10,10))
plt.plot(time,flux, color = '#FCCC25',marker = '.',linestyle ='None',label = 'Flux with noise')
plt.plot(time,flux_maxl, color = '#E06461', marker = 'None', label = 'Max likelihood flux')
plt.plot(time,flux_true, color = '#6A00A7', marker = 'None', label = 'True Flux')
plt.legend(fontsize=10)
plt.xlabel("Time in Days",fontsize=14)
plt.ylabel("Thermal Flux",fontsize=14)
plt.plot()


# In[35]:


L = 1
C = changeOfBasis(L)
time = np.linspace(-0.1, 0.1, 1000)


# In[36]:


# Initialize around the max likelihood solution
T_pos = T_maxl + 1e-4 * np.random.randn(32,4)
nwalkers,ndim = T_pos.shape

# Run an MCMC chain
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args = (params,time,flux,flux_err,star,planet,L,C))
sampler.run_mcmc(T_pos, 5000, progress=True)                                


# In[52]:


fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r"$T_1$", r"$T_2$", r"$T_3$",r"$T_4$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], alpha=0.3,lw = 1)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Number of Steps");


# In[53]:


# An estimate of the integrated autocorrelation time
tau = sampler.get_autocorr_time()
print(tau)


# In[54]:


flat_samples = sampler.get_chain(discard=100, thin=30, flat=True)
print(flat_samples.shape)


# In[57]:


labels = ["T_1","T_2","T_3","T_4"]
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))


# In[58]:


T_r = np.array([0.703,0.339,-0.001,-0.184])
X_r = np.dot(C,T_r)

flux_r = getLightcurve(X_r,params,time,star,planet,L)


# In[59]:


plt.figure(figsize=(10,10))
plt.plot(time,flux, color = '#FCCC25',marker = '.',linestyle ='None',label = 'Flux with noise')
plt.plot(time,flux_maxl, color = '#E06461', marker = 'None', label = 'Max likelihood flux')
plt.plot(time,flux_r, color = '#B02A8F', marker = 'None', label = 'MCMC result')
plt.plot(time,flux_true, color = '#6A00A7', marker = 'None', label = 'True Flux')
plt.legend(fontsize=10)
plt.xlabel("Time in Days",fontsize=14)
plt.ylabel("Thermal Flux",fontsize=14)
plt.plot()


# In[ ]:




