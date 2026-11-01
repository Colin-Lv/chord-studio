"""Root-relative learned chord classifier. No key or reference inputs."""
import numpy as np


def features(tones, bass, tone_activity, bass_activity):
    tones, bass = np.asarray(tones), np.asarray(bass)
    return np.asarray([np.r_[np.roll(tones,-r),np.roll(bass,-r),
                            np.roll(tones,-r)/max(tones.max(),1e-9),
                            tone_activity,bass_activity] for r in range(12)],dtype=np.float32)


def predict(model, tones, bass, tone_activity, bass_activity):
    raw=model.predict_proba(features(tones,bass,tone_activity,bass_activity))
    probability=np.full((12,10),1e-9)
    for j,c in enumerate(model.classes_):
        if c: probability[:,int(c)-1]=raw[:,j]
    probability=probability.reshape(-1)
    return probability/probability.sum()
