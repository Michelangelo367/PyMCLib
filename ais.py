import functions as fn
import numpy as np
from scipy.stats import multivariate_normal as mvn


class pmc:
    """ Population Monte Carlo class
    
    class properties:
        N: int: number of populations
        D: int: dimension of sampling space
        K: int: number of samples per-population
        mu: N*D nparray: population means
        C: D*D nparray: covariance of proposal distribution
        rho: float: tempering of target distribution
        resample_method: 'global' or 'local'
        x: N*K*D nparray: samples
        w: N*K nparray: sample weights
        logp = logtarget(x) : function: log target distribution
            x: M*D, nparray
            logp: M, nparray
    """

    def __init__(self, mu0, K, logtarget):
        """ construction funciton
        inputs:
            mu0: N*D, nparray: initial means, also defines number and dimension of populations
            K: int: number of samples per population
            logp = logtarget(x) : function: log target distribution
                x: M*D, nparray
                logp: M, nparray
        """
        # passing the parameters
        self.mu = mu0
        self.N, self.D = np.shape(mu0)
        self.K = K
        self.logtarget = logtarget
        # default parameters
        self.C = np.eye(self.D)
        self.rho = 1.0
        self.resample_method = "global"
        # initial sampling
        self.x = np.zeros(shape=(self.N, self.K, self.D))  # space allocation
        self.w = np.zeros(shape=(self.N, self.K))  # space allocation
        for n in range(self.N):
            self.x[n, :, :] = mvn.rvs(mean=self.mu[n, :], cov=self.C, size=self.K)
            self.w[n, :] = np.ones(shape=self.K) / self.K
        return

    def setSigma(self, sig):
        """ set the proposal covariance by the shared std of all dimension
        inputs:
            sig: float: std of all dimensions
        """
        self.C = np.eye(self.D) * (sig ** 2)
        return

    def setRho(self, rho):
        """ set target distribution tempering
        inputs:
            rho: float: tempering the target by pi(.)^rho
        """
        self.rho = rho
        return

    def sample(self):
        """ One iteration if the sampling procesdure
        outputs:
            outx: M*D nparray: sample locations of current iteration
            outlogw: M nparray: sample log weights of current iteration
        """
        # log weights of samples for current population
        logw_n = np.ones([self.N, self.K])
        # log tempered weights of samples for current population
        logTw_n = np.ones([self.N, self.K])
        for n in range(self.N):  # for each population
            logprop = np.ones([self.K])  # log proposal probability
            for k in range(self.K):  # for each particle in the population
                self.x[n, k, :] = mvn.rvs(
                    mean=self.mu[n, :], cov=self.C, size=1
                )  # sampling from proposal
                logprop[k] = fn.logmean(
                    mvn.logpdf(x=self.mu, mean=self.x[n, k, :], cov=self.C)
                )  # DM-weights
            # weights
            logw_n[n, :] = self.logtarget(self.x[n, :, :]) - logprop
            # tempered witghts
            logTw_n[n, :] = self.logtarget(self.x[n, :, :]) * self.rho - logprop
        # prepare global particles for output
        outx = np.reshape(self.x, (-1, self.D))
        outlogw = np.reshape(logw_n, (-1))
        outlogTw = np.reshape(logTw_n, (-1))
        # resampling
        if self.resample_method == "global":
            ind = np.random.choice(
                a=np.arange(self.N * self.K), p=fn.logw2w(outlogTw), size=self.N
            )
            self.mu = outx[ind, :]
        elif self.resample_method == "local":
            for n in range(self.N):
                ind = np.random.choice(a=np.arange(self.K), p=fn.logw2w(logTw_n[n, :]))
                self.mu[n, :] = self.x[n, ind, :]
        else:
            print("wrong resample type")
        return outx, outlogw