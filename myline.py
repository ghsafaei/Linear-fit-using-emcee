


import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl



# Choose the "true" parameters.
m_true =-0.9594
b_true = 4.294
f_true = 1.001


# Generate some synthetic data from the model.
N = 10
yerr = 0.5
#err =0.4



input=np.genfromtxt('/home/gh/Desktop/MCMC/3---/input')
x=(input[:,0])
y=(input[:,1])



# Define the probability function as likelihood * prior.




def lnprior(theta):
    m, b, lnf = theta
    if -1.0 < m < 1.5 and 0.0 < b < 1.0 and -1.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))





nll = lambda *args: -lnlike(*args)

result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))




m_ml, b_ml, lnf_ml = result["x"]






def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)




# Find the maximum likelihood value.
#import scipy.optimize as op





# Set up the sampler.
ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

sampler.run_mcmc(pos, 500)



# Compute the quantiles.
#samples[:, 2] = np.exp(samples[:, 2])


samples=sampler.flatchain


m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))


#pl.plot(x,y)
#pl.show()

print("""MCMC result:
    m = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    b = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    f = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(m_mcmc, m_true, b_mcmc, b_true, f_mcmc, f_true))
