from pathlib import Path
import sys,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from full_common import HERE,ROOT,config,load_data,selections,score_vectors,parquet_write,atomic_json
sys.path.insert(0,str(ROOT/'src'))
from models import classical
from run_loss_controls import log_gaussian

def main(key):
    d=load_data(key);cfg=config();ids=d['target'];out=HERE/'results'/key/'classical';out.mkdir(parents=True,exist_ok=True)
    protocols=[('nested',b) for b in cfg['budgets']]+[('default',0)]
    methods=['gaussian','log_gaussian','skew_gaussian','log_skew_gaussian','spline','pchip']
    for protocol,B in protocols:
      for start in range(0,len(ids),cfg['cpu_chunk']):
        chunk=ids[start:start+cfg['cpu_chunk']];file=out/f'{protocol}_B{B}_{start:07}.parquet'
        if file.exists():continue
        fit,test,val=selections(d,chunk,protocol,B);frames=[]
        for method in methods:
            pred=np.zeros((len(chunk),d['steps'].shape[1]),np.float32);fallback=[]
            for j,i in enumerate(chunk):
                n=int(d['length'][i]);obs=np.union1d(fit[j],val[j]) if protocol=='default' else fit[j]
                grid=d['steps'][i,:n];x=grid[obs];y=d['frequency'][i,obs]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',OptimizeWarning)
                    if method.startswith('log_'):p,f=log_gaussian(x,y,grid,method=='log_skew_gaussian')
                    else:p,f=classical(method,x,y,grid)
                pred[j,:n]=p;fallback.append(f)
            row=pd.DataFrame(score_vectors(d,chunk,pred,test,[np.union1d(a,v) for a,v in zip(fit,val)] if protocol=='default' else fit))
            row['method']=method;row['seed']=0;row['protocol']=protocol;row['budget']=B;row['fallback']=fallback
            frames.append(row)
        parquet_write(file,pd.concat(frames,ignore_index=True))
        atomic_json(out/'progress.json',dict(status='running',protocol=protocol,budget=B,completed_through=start+len(chunk),target_curves=len(ids),updated=time.time()))
        print(f'{key} {protocol} B{B} classical {start+len(chunk)}/{len(ids)}',flush=True)
    atomic_json(out/'complete.json',dict(status='complete',target_curves=len(ids),methods=methods,protocols=protocols,full_target_coverage=True))

if __name__=='__main__':main(sys.argv[1])
