import numpy as np
import sys
from pybasicbayes.distributions import Gaussian
from gen_synthetic import generate_data

if __name__ == '__main__':
    prefix = 'cy'
    K=5
    A = np.array([[.00, .99,0.00, .00, .01],
                  [.01, .00, .99, .00, .00],
                  [.00, .01, .00, .99, .00],
                  [.00, .00, .01, .00, .99],
                  [.99, .00, .00, .01, .00]])
    means= [(0, 0), (10, 10), (10, 10), (0, 0), (10, 10)]
    means = [Gaussian(mu=np.array(m), sigma=np.eye(2)) for m in means]
    means = np.array(means)
    obs, sts, _ = generate_data(A, means, 10000)
    np.savetxt('data/%s_data.txt' % prefix, obs)
    np.savetxt('data/%s_sts.txt' % prefix, sts)
