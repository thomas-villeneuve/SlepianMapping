#!/usr/bin/env python
# coding: utf-8

# In[1]:


import starry
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from IPython.display import display, Math


# In[2]:


# Compute lightcurves of order one Slepian functions
time = np.linspace(-0.1, 0.1, 1000)
star = starry.kepler.Primary()
s1 = starry.kepler.Secondary(lmax = 1)
s1[1,0] = np.sqrt(2)/2
s1.porb = 1
s1.prot = 1
s1.a = 5
s1.tref = 0.5
s1.inc = 87
s1.L = 0.01
systems1 = starry.kepler.System(star, s1)
systems1.compute(time)
F1 = s1.lightcurve

s2 = starry.kepler.Secondary(lmax = 1)
s2[1,-1] = 1
s2.prot = 1
s2.a = 5
s2.tref = 0.5
s2.inc = 87
s2.L = 0.01
systems2 = starry.kepler.System(star, s2)
systems2.compute(time)
F2 = s2.lightcurve

s3 = starry.kepler.Secondary(lmax = 1)
s3[1,1] = 1
s3.prot = 1
s3.a = 5
s3.tref = 0.5
s3.inc = 87
s3.L = 0.01
systems3 = starry.kepler.System(star, s3)
systems3.compute(time)
F3 = s3.lightcurve

s4 = starry.kepler.Secondary(lmax = 1)
s4[1,-1] = -np.sqrt(2)/2
s4.prot = 1
s4.a = 5
s4.tref = 0.5
s4.inc = 87
s4.L = 0.01
systems4 = starry.kepler.System(star, s4)
systems4.compute(time)
F4 = s4.lightcurve


# In[7]:


np.random.seed(264)

# Create an arbitray star-planet system
planet1 = starry.kepler.Secondary(lmax = 1)
planet1[1,0] = 0.7 *np.sqrt(0.5)
planet1[1,-1] = 0.3   
planet1.porb = 1
planet1.prot = 1
planet1.a = 5
planet1.tref = 0.5
planet1.inc = 87
planet1.L = 0.01
planet1.show()

# Compute the planet lightcurve
system1 = starry.kepler.System(star, planet1)
system1.compute(time)
flux = planet1.lightcurve

# Generate some synthetic data
yerr = 0.001
flux += yerr * np.random.randn(1000)


# In[17]:


# Plot the lightcurve and synthetic data
plt.figure(figsize=(8,8))
plt.plot(time,0.7*F1 + 0.3*F2, 'r-',label = 'lightcurve')
plt.plot(time,flux,'mo', markersize=2,label = 'data with noise')
plt.xlabel("Time [Days]", fontsize = 16)
plt.ylabel('Planet Flux', fontsize = 16)
plt.legend()


# In[8]:


# Define the likelihood function
def log_likelihood(theta, time, flux):
    a,b,c,d,s = theta
    model = a*F1 + b*F2 +c*F3 + d*F4
    sigma2 = s ** 2 
    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

# Find the numerical optimum of the likelihood function
nll = lambda *args: -log_likelihood(*args)
initial = np.array([0.1,0.1,0.1,0.1,0.1])
soln = minimize(nll, initial, args = (time, flux))
a_ml, b_ml, c_ml, d_ml, s_ml = soln.x

print("a = {0:.3f}".format(a_ml))
print("b = {0:.3f}".format(b_ml))
print("c = {0:.3f}".format(c_ml))
print("d = {0:.3f}".format(d_ml))
print("s = {0:.3f}".format(s_ml))


# In[9]:


# Define the full log posterior probability function
def log_prior(theta):
    a,b,c,d,s = theta
    if -1 < a < 1 and -1 < b < 1 and -1 < c < 1 and -1 < d < 1 and 0 < s < 1:
        return 0.0
    return -np.inf

def log_probability(theta, time, flux):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, time, flux)


# In[10]:


# Run 5,000 steps of MCMC
import emcee
pos = soln.x + 1e-4 * np.random.randn(32, 5)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(time, flux))
sampler.run_mcmc(pos, 5000, progress=True)


# In[11]:


# An estimate of the integrated autocorrelation time
tau = sampler.get_autocorr_time()
print(tau)


# In[12]:


labels = ["a", "b", "c", 'd','s']
flat_samples = sampler.get_chain(discard=100, thin=30, flat=True)
print(flat_samples.shape)


# In[13]:


# Corner plot
fig = corner.corner(flat_samples, labels=labels, truths=[0.7, 0.3,0,0,0.001])


# In[14]:


# Display the results:
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))


# In[19]:


# Plot the true lightcurve, synthetic data and the recovered lightcurve
plt.figure(figsize=(8,8))
plt.plot(time,flux,'mo', markersize=2, label = 'Data with Noise')
plt.plot(time,0.7*F1 + 0.3*F2, 'r--',label = 'True lightcurve')
plt.plot(time, a_ml * F1 + b_ml*F2 +c_ml*F3 + d_ml*F4, 'g-', label = 'Recovered lightcurve')
plt.xlabel("Time [Days]", fontsize = 16)
plt.ylabel('Planet Flux', fontsize = 16)
plt.legend()

