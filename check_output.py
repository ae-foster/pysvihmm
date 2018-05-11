import numpy as np
import cPickle as pkl
import sys

assert len(sys.argv) == 3
prefix = sys.argv[1]
par = sys.argv[2]

fname = 'data/%s_results/%s_par%s_fold0.pkl' % (prefix, prefix, par)
with open(fname, 'rb') as f:
    hmm_obj = pkl.loads(f.read())

A = hmm_obj.var_tran.copy()
A /= A.sum(axis=1)[:, np.newaxis]
print(np.round(A, decimals=2))

for distn in hmm_obj.var_emit:
    print(distn.mu)

print(np.mean(hmm_obj.var_x, axis=0))
