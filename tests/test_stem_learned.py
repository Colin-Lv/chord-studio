import numpy as np
from app.web.stem_learned import features,predict


class Model:
    classes_=np.arange(11)
    def predict_proba(self,x):
        values=np.column_stack([np.ones(len(x)),np.maximum(x[:,:10],.001)])
        return values/values.sum(axis=1,keepdims=True)


def test_transposing_audio_rotates_roots_not_quality():
    rng=np.random.default_rng(1);t=rng.random(12);b=rng.random(12)
    a=predict(Model(),t,b,1,1).reshape(12,10)
    c=predict(Model(),np.roll(t,3),np.roll(b,3),1,1).reshape(12,10)
    np.testing.assert_allclose(c,np.roll(a,3,axis=0))


def test_silent_features_are_finite():
    assert np.isfinite(features(np.zeros(12),np.zeros(12),0,0)).all()
    probability=predict(Model(),np.zeros(12),np.zeros(12),0,0)
    assert probability.shape==(120,)
    np.testing.assert_allclose(probability.sum(),1)
