import numpy as np
import soundfile as sf
from app.web import strings_runner as runner

def check_kernel(tmp_path):
    source=tmp_path/'input.wav';x=np.random.default_rng(42).normal(0,.1,(8192,2)).astype('float32');sf.write(source,x,44100,subtype='FLOAT')
    calls=[]
    def predict(audio,*args):
        calls.append(audio.copy())
        if len(calls)==1:return {'strings':audio*.25,'other':audio*.9}
        return {k:audio/6 for k in ['vocals','drums','bass','other','piano','guitar']}
    from unittest.mock import patch
    out=tmp_path/'out'
    with patch.object(runner,'predict',predict):runner.separate(source,out,'v2',tmp_path,tmp_path)
    np.testing.assert_allclose(calls[1],x*.75,atol=1e-8)
    y,_=sf.read(out/'strings.wav');np.testing.assert_array_equal(y,x*.25)
    total=sum(sf.read(out/(k+'.wav'))[0] for k in ['strings','vocals','drums','bass','other','piano','guitar'])
    np.testing.assert_allclose(total,x,atol=5e-8)


def test_second_stage_reads_only_complement_and_keeps_target(tmp_path):
    check_kernel(tmp_path)

if __name__=="__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:check_kernel(Path(folder))
    print("Complement routing and unchanged target verified")
