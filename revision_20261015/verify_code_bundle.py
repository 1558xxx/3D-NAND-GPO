"""Verify copied runtime provenance and optional synthetic CPU compatibility only."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'CODE_PROVENANCE.json').read_text(encoding='utf8'))
    for r in manifest['files']:
        p=ROOT/r['path'];assert p.is_file(),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
    files=list(ROOT.rglob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    print('Copied source hashes:',len(manifest['files']),'Python syntax:',len(files),'passed')
    if '--runtime' not in sys.argv:return
    sys.path.insert(0,str(ROOT/'full_physical_run_20261014'))
    import numpy as np
    import classical_worker,gpu_worker,generator_cost_retained,finalize_retained_run,delivery_statistics
    from queue_scope import retained_scope
    from gpu_ops import new_vector,predict
    from full_common import fixed_test,query_order
    assert retained_scope()
    v=new_vector(42);assert v.shape==(609,)
    x=np.linspace(-1,1,20,dtype=np.float32);y=predict(v[None,:],x[None,:],device='cpu');assert np.isfinite(y).all() and (y>=0).all()
    test=fixed_test(len(x));order=query_order(x,test)
    assert not set(test)&set(order[:9]) and set(order[:5])<=set(order[:7])<=set(order[:9])
    from TimeTransformer import Transformer3
    n=Transformer3(d_input=1,d_model=16,d_output=1,num_wl=256,N=4,layernum=609,dropout=.1,pe='original')
    assert sum(p.numel() for p in n.parameters())==15921
    grid=np.arange(20,dtype=float);truth=20+100*np.exp(-.5*((grid-10)/3)**2)
    prediction,_=classical_worker.log_gaussian(grid,truth,grid)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    print('Runtime imports, original architecture, MLP prediction, nested protocol and log-Gaussian synthetic checks passed')
    print('Synthetic compatibility checks do not replace or claim a full scientific rerun.')
if __name__=='__main__':main()
