import numpy as np
import cPickle as pkl
import sys

assert len(sys.argv) == 3
prefix = sys.argv[1]
par = sys.argv[2]

fname = 'data/%s_results/%s_par%s_fold0.pkl' % (prefix, prefix, par)
with open(fname, 'rb') as f:
    hmm_obj = pkl.loads(f.read())

print 'Reuse messages:', hmm_obj.reuseMsg, 'Corrected trans:', hmm_obj.correctTrans
print 'L', 2*hmm_obj.metaobs_half+1, 'M', hmm_obj.mb_sz
print 'Learning rates: tau', hmm_obj.tau, 'kappa', hmm_obj.kappa
print 'A'
A = hmm_obj.var_tran.copy()
A /= A.sum(axis=1)[:, np.newaxis]
print(np.round(A, decimals=2))

print 'Cluster means'
for distn in hmm_obj.var_emit:
    print(np.round(distn.mu, decimals=0))

print 'State marginal'
print(np.round(np.mean(hmm_obj.var_x, axis=0), decimals=2))
