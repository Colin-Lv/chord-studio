import itertools
import unittest
import numpy as np
from scipy.special import logsumexp
from app.chord_crf_diagnostics import marginals, maximum_path


class CRFDiagnosticsTests(unittest.TestCase):
    def test_matches_exhaustive_paths_including_initial_sentinel(self):
        rng=np.random.default_rng(18)
        for n in [1,3]:
            for k in [2,3]:
                e=rng.normal(size=(n,k));pi=rng.normal(size=k);a=rng.normal(size=(k,k));tau=rng.normal(size=k)
                paths=list(itertools.product(range(k),repeat=n+1))
                scores=np.array([pi[y[0]]+sum(a[y[t],y[t+1]]+e[t,y[t+1]] for t in range(n))+tau[y[-1]] for y in paths])
                z=logsumexp(scores);expected=np.zeros_like(e)
                for path,p in zip(paths,np.exp(scores-z)):
                    for t,state in enumerate(path[1:]):expected[t,state]+=p
                actual,actual_z=marginals(e,pi,a,tau)
                np.testing.assert_allclose(actual,expected,atol=1e-12)
                self.assertAlmostEqual(actual_z,z)
                self.assertEqual(tuple(maximum_path(e,pi,a,tau)),paths[int(scores.argmax())][1:])

    def test_constant_potential_shift_does_not_change_marginals(self):
        e=np.array([[.2,-.4],[.8,.5]])
        pi=np.array([.1,.2]);a=np.array([[2.,-.8],[.1,1.]]);tau=np.zeros(2)
        p,_=marginals(e,pi,a,tau);q,_=marginals(e+10000,pi,a,tau)
        np.testing.assert_allclose(p,q,atol=1e-10)


if __name__=='__main__':unittest.main()
