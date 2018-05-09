import numpy as np
import cPickle as pkl

prefix='rc'
fname = 'data/%s_results/%s_par0_fold0.pkl' % (prefix, prefix)
with open(fname, 'rb') as f:
    hmm_obj = pkl.loads(f.read())

A = hmm_obj.var_tran.copy()
A /= A.sum(axis=1)[:, np.newaxis]
print(np.round(A, decimals=2))

for distn in hmm_obj.var_emit:
    print(distn.mu)

print(np.mean(hmm_obj.var_x, axis=0))
