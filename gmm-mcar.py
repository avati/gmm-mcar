import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial

np.random.seed(230)

K = 3

##
## For diagnostics. Check if covariance matrix is singular.
##
def is_singular(cov):
    d = cov.shape[0]
    if d > 0:
        vec = np.zeros(d)
        try:
            p = mvn.logpdf(vec, mean=vec, cov=cov)
        except:
            return True
    return False


##
## Create a random PSD matrix of dimension `d`
##
def randpsd(d):
    A = np.random.normal(0, 1, size=(d, d))
    return A.dot(A.T)


##
## Extract a square sub-matrix with indices provided in `id`
##
def submat(M, id):
    # Extract submatrix. id is an array of integers
    return M[np.ix_(id, id)]

##
## Generate synthetic data with missingness represented as np.NaN
##
## N - number of examples
## D - dimension of each example
## K - number of clusters
##
def gen_data(K, missing=0.15, N=7, D=3):
    phi = np.random.rand((K))
    phi /= np.sum(phi)
    data = np.zeros((N, D))
    labels = np.zeros(N)

    mu = np.random.normal(0, 1, size=(K, D))
    s2 = [ randpsd(D) for _ in range(K) ]

    for i in range(N):
        z = np.random.choice(K, p=phi)
        x = mvn.rvs(mu[z], s2[z], size=1)
        data[i, :] = x
        labels[i] = z

    while True:
        mask = np.random.choice([0, 1], size=(N, D), p=[missing, 1 - missing])
        if np.max(np.sum(mask == 0, axis=1)) < D:
            break
    data[mask == 0] = np.nan
    return data, labels

##
## Calculate the posterior `p(y|x,z)`. Here `y` is the missing
## components of the data, `x` is the non-missing components
## of the data, and `z` is the cluster identity.
##
def calc_y_xz(d, mu, s2):
    idx = np.argwhere(np.isnan(d) == 0).reshape(-1)
    nan = np.argwhere(np.isnan(d)).reshape(-1)
    s2yy = s2[np.ix_(nan, nan)]
    s2xx = s2[np.ix_(idx, idx)]
    s2xy = s2[np.ix_(idx, nan)]
    s2yx = s2[np.ix_(nan, idx)]
    mu_x = mu[idx]
    mu_y = mu[nan]
    d_x = d[idx]
    d_y = d[nan]

    mu_y_x = mu_y + s2yx.dot(np.linalg.inv(s2xx)).dot(d_x - mu_x)
    s2_y_x = s2yy - s2yx.dot(np.linalg.inv(s2xx)).dot(s2xy)

    return (mu_y_x, s2_y_x)

def E_step(data, params):
    # log-likelihood of the observed data
    phi, mu, s2 = params
    N, D = data.shape
    K = len(mu)
    logZ = np.zeros((N, K))

    logP_z = np.zeros(K) # logP(z)
    logP_x_z = np.zeros((N, K)) # logP(x|z)
    logP_xz = np.zeros((N, K)) # logP(x,z)
    logP_x = np.zeros(N) # logP(x)
    y_xz = { } # y|x,z : mu and s2 of posterior

    for k in range(K):
        logP_z[k] = np.log(phi[k])
        ##
        ## !!! THIS LOOP KILLS COMPUTATIONAL PERFORMANCE !!!
        ##
        for i, d in enumerate(data):
            idx = np.argwhere(np.isnan(d) == 0).reshape(-1)

            logP_x_z[i, k] = mvn.logpdf(d[idx], mean=mu[k][idx], cov=submat(s2[k], idx))
            logP_xz[i, k] = logP_x_z[i, k] + logP_z[k]

            y_xz[(i, k)] = calc_y_xz(d, mu[k], s2[k])

    logP_x = logP_xz[:, 0]
    for k in range(1, K):
        logP_x = np.logaddexp(logP_x, logP_xz[:, k])

    logP_z_x = logP_xz - logP_x.reshape((-1, 1))

    return (logP_z_x, y_xz, logP_x)

def M_step(data, logP_z_x, y_xz):
    (N, K) = logP_z_x.shape
    P_z_x = np.exp(logP_z_x)

    ##
    ## !!! THIS LOOP KILLS COMPUTATIONAL PERFORMANCE !!!
    ##
    def fill_DZ(data, y_xz, k):
        (N, D) = data.shape
        Z = np.zeros((N, D, D))
        D = np.copy(data)
        for (i, d) in enumerate(data):
            nan = np.argwhere(np.isnan(d)).reshape(-1)
            mu_y_x, s2_y_x = y_xz[(i, k)]
            D[i, nan] = mu_y_x
            Z[i][np.ix_(nan, nan)] = s2_y_x
        return D, Z

    def s2_k(data, mu, w, k):
        D, Z = fill_DZ(data, y_xz, k)
        d = D - mu
        s2 = np.sum((d[:, None, :] * d[:, :, None] + Z) * w, axis=0) / np.sum(w)

        return s2

    def mu_k(data, w, k):
        D, _ = fill_DZ(data, y_xz, k)
        return np.sum(D * w, axis=0) / np.sum(w)

    mu = [ mu_k(data, P_z_x[:, k:k+1], k) for k in range(K) ]
    s2 = [ s2_k(data, mu[k], P_z_x[:, k:k+1, None], k) for k in range(K) ]
    phi = np.mean(np.exp(logP_z_x), axis=0)

    return (phi, mu, s2)

def fit_em(data, K):
    # Randomly initialize parameters
    N, D = data.shape
    phi = np.array([1./K] * K)

    # impute with column means, just for initialization
    impute = np.copy(data)
    col_mean = np.nanmean(impute, axis=0)
    inds = np.where(np.isnan(impute))
    impute[inds] = np.take(col_mean, inds[1])

    group = np.random.choice(K, N)
    mu = [np.mean(impute[group == k, :], axis=0) for k in range(K)]
    s2 = [np.cov(impute[group == k, :].T) for k in range(K)]

    params = (phi, mu, s2)

    ll_prev = None

    while True:
        logP_z_x, y_x, logP_x = E_step(data, params)

        params = M_step(data, logP_z_x, y_x)

        ll = logP_x.sum()
        print('logP(x) =', ll)
        if ll_prev:
            if ll < ll_prev:
                print('BUG!!!')
                break
            if ll - ll_prev < 1e-3:
                print('Converged!')
                break
        ll_prev = ll

    return params

def main():
    data, _ = gen_data(K, missing=0.2, N=100, D=3)

    phi, mu, s2 = fit_em(data, K)

if __name__ == '__main__':
    main()

