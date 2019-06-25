import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


# Sample related --------------------------------------


def logw2w(logw):
    """ convert log weightd to normalized weights
    inputs:
        logw: M nparray: log weights
    return:
        w: M nparray: normalized weights
    """
    logw = logw - np.max(logw)
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    return w


def logmean(logw):
    """ return the log of mean of weights given the log weights
    inputs:
        logw: M nparray: log weights
    return:
        return: float: log of mean of weights
    """
    log_scale = np.max(logw)
    logw = logw - log_scale
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    log_scale = log_scale + np.log(sumw)
    m = np.mean(w)
    return np.log(m) + log_scale


def weightedsum(x, w):
    """ Weighted sum of vectors
    inputs:
        x: M*D, nparray: samples
        w: M, nparray: weights
    return:
        accu: D, nparray: weighted sum of samples
    """
    accu = np.ones(np.size(x, 1)) * 0
    for i in range(len(w)):
        accu += w[i] * x[i, :]
    return accu


def mvnfit(x, w):
    """ fit weighted samples by a Multivariable Gaussian
    inputs:
        x: M*D, nparray: samples
        w: M, nparray: weights
    return:
        mu: D nparray: mvn mean
        C: D*D nparray: mvn covariance
    """
    w = w / np.sum(w)
    mu = weightedsum(x, w)
    x_mu = x - mu
    C = x_mu.transpose() @ np.diag(w) @ x_mu
    return mu, C


def plotsamples(x, w, dim=[0,1]):
    """ plot weighted samples in 2D
    inputs:
        x: M*D, nparray: samples, D>=2
        w: M, nparray: weights
        dim: list of int: dim[0], and dim[1] are the two dimensions to display
    """
    alpha = w / np.max(w)
    color = np.array([0.8, 0.2, 0.6])
    rgba = np.zeros((len(w), 4))
    for i in range(len(w)):
        rgba[i, 0:3] = color
        rgba[i, 3] = alpha[i]
    plt.scatter(x[:, dim[0]], x[:, dim[1]], c=rgba, marker="o", s=100)


# Target distributions ------------------------------------------------------


def logbanana(x, D, a=1.0, b=2.0):
    """ log banana distribution, not normalized
    inputs:
        x: M*D nparray: input data
        D: int: data dimension
        a: float: center parameter
        b: float: sharpness parameter
    return:
        logp: M, nparray: likelihood of data
    """
    p = 0
    for d in range(D - 1):
        p += b * (x[:, d + 1] - x[:, d] ** 2) ** 2 + (a - x[:, d]) ** 2
    return -p


def lognormal(x, D, mu=None, C=None):
    """ log normal distribution, normalized
    inputs:
        x: M*D nparray: input data
        D: int: data dimension
        a: float: center parameter
        b: float: sharpness parameter
    return:
        logp: M, nparray: likelihood of data
    """
    if mu is None:
        mu = np.zeros(shape=D)
    if C is None:
        C = np.eye(D)
    logp = mvn.logpdf(x, mean=mu, cov=C)
    return logp