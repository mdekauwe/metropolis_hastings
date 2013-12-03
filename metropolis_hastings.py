#!/usr/bin/env python
"""
Use a Metropolis-Hastings approach to fit a two parameter straight line. 
Simples?

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (02.12.2013)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class MetropolisHastings(object):
    def __init__(self, num_samples, burn_in, X, obs):
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.X = X
        self.obs = obs
        
    def sample(self, params, tune_sd):
        # initialise the chain
        self.chain = np.zeros((self.num_samples, len(params)))
        self.chain[0,:] = params    
        
        # grabs some random numbers for the metropolis ratio
        logu = np.log(np.random.uniform(0.0, 1.0, self.num_samples))
        
        for i in xrange(1, self.num_samples):
            # propose a new candidate
            candidate = self.proposal(self.chain[i-1,:], tune_sd)
            
            # do we accept the proposed candidate?
            if logu[i] < self.alpha(candidate, self.chain[i-1,:]):
                # accept the candiate
                self.chain[i,:] = candidate
            else: 
                # reject the candiate
                self.chain[i,:] = self.chain[i-1,:]
        return self.chain[self.burn_in:self.num_samples] 
        
    def alpha(self, candidate, old_candidate):
        """ Acceptance probability """
        return (self.posterior_distribution(candidate) - 
                self.posterior_distribution(old_candidate))
        
    def proposal(self, params, sigma):
        return stats.norm.rvs(loc=params, scale=sigma, size=len(params))
        
    def likelihood(self, params):
        
        pred = model(params, self.X)
        sigma = params[2]
        return sum(stats.norm.logpdf(x=self.obs, loc=pred, scale=sigma))  

    def prior_distribution(self, params):
        """ Use uninformative uniform priors """
        intercept = params[0]
        slope = params[1]
        sigma = params[2]
        
        intercept_prior = stats.norm.logpdf(intercept, loc=0.0, scale=10.0)
        #intercept_prior = stats.uniform.logpdf(intercept, loc=0.0, scale=50.0)
        slope_prior = stats.uniform.logpdf(slope, loc=0.0, scale=15)
        sigma_prior = stats.uniform.logpdf(sigma, loc=0.0, scale=30.0)
        return intercept_prior + slope_prior + sigma_prior

    def posterior_distribution(self, params):
        return self.likelihood(params) + self.prior_distribution(params)

def model(params, X):
    intercept = params[0]
    slope = params[1]
    
    return intercept + slope * X
     
if __name__ == "__main__":
    
    #------------------------
    # Generate some fake data
    #------------------------
    n_samples = 100
    param_truth = [16.0, 3.4, 10.8] #intercept, slope, sd
    X = np.arange(n_samples)
    
    # Observation model is y = mx + c + N(0, sigma)
    obs = model(param_truth, X) +\
         np.random.normal(loc=0.0, scale=param_truth[2], size=len(X))
    
    
    params_guess = [0.0, 8.4, 0.5]
    tune_sd = 0.08 # need to play with this to get a decent acceptance rate
    
    #------------------------
    #   Fit the parameters 
    #------------------------
    num_samples = 10000
    burn_in = 0.1 * num_samples
    test_tune = True
    
    if test_tune == True:
        # We need to tune the proposal matrix, so walk around parameter space
        # this is basically unchanged from what Jose sent me
        print 'Aim for 20-30 %'
        MC_test = MetropolisHastings(num_samples, burn_in, X, obs)
        for tune in np.linspace(0.001, 10.0, 10.0):
            chain = MC_test.sample(params_guess, tune)
            accepted = len(np.unique(chain[:,0]))
            accepted = (accepted/(num_samples-burn_in)*100.)
            print "Step size: %6.4g ----> acceptance rate: %6.2f %%" % \
                    (tune, accepted)
    
    MC = MetropolisHastings(num_samples, burn_in, X, obs)
    chain = MC.sample(params_guess, tune_sd)
    
    print 'Aim for 20-30 %'
    accepted = len(np.unique(chain[:,0]))
    print "Acceptance: %0.2f %%" % (accepted/(num_samples-burn_in)*100.)
    
    
    #------------------------
    #      Make Plots
    #------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, obs, "ro", label="Obs")
    params_fit = [np.mean(chain[:,0]), np.mean(chain[:,1])]
    YY = model(params_fit, X) 
    ax1.plot(X, YY, "b-", label="Fit")
    ax1.legend(numpoints=1, loc='best', shadow=True).draw_frame(True)
    fig.savefig('/Users/mdekauwe/Desktop/model_fit.png', dpi=150)
    
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.8)
    
    ax1 = fig.add_subplot(311)
    n, bins, patches = ax1.hist(chain[:,0], bins=50, normed=False)
    ax1.axvline(param_truth[0], 0, n.max(), color='r', linestyle='--', 
                linewidth=2)
    ax1.set_ylabel("Frequency")
    ax1.set_title("Posterior Intercept")
    
    ax2 = fig.add_subplot(312)
    n, bins, patches = ax2.hist(chain[:,1], bins=50, normed=False)
    ax2.axvline(param_truth[1], 0, n.max(), color='r', linestyle='--', 
                linewidth=2)
    ax2.set_ylabel("Frequency")
    ax2.set_title("Posterior Slope")
    
    ax3 = fig.add_subplot(313)
    n, bins, patches = ax3.hist(chain[:,2], bins=50, normed=False)
    ax3.axvline(param_truth[2], 0, n.max(), color='r', linestyle='--', 
                linewidth=2)
    ax3.set_xlabel('Parameter range')
    ax3.set_ylabel("Frequency")
    ax3.set_title("Posterior sigma")
    fig.savefig('/Users/mdekauwe/Desktop/parameter_PDFs.png', dpi=150)
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.8)
    
    ax1 = fig.add_subplot(311)
    ax1.plot(chain[:,0], color='r', linestyle='-', label='Intercept')
    ax1.set_xlabel("Iterations")
    ax1.legend(numpoints=1, loc='best', shadow=True).draw_frame(True)
    ax1.set_title("Trace: Intercept")
    
    ax2 = fig.add_subplot(312)
    ax2.plot(chain[:,1], color='g', linestyle='-', label='Slope')
    ax2.set_xlabel("Iterations")
    ax2.legend(numpoints=1, loc='best', shadow=True).draw_frame(True)
    ax2.set_title("Trace: Slope")
    
    ax3 = fig.add_subplot(313)
    ax3.plot(chain[:,2], color='g', linestyle='-', label='Sigma')
    ax3.set_ylabel('Parameter range')
    ax3.set_xlabel("Iterations")
    ax3.set_title("Trace: obs sigma")
    
    fig.savefig('/Users/mdekauwe/Desktop/parameter_trace.png', dpi=150)
    plt.show()
    
    
    