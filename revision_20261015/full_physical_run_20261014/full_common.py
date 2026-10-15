from pathlib import Path
import json,time,os,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for k in range(20):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.1*(k+1))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read_json(path,default=None):
    for k in range(5):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except (PermissionError,json.JSONDecodeError):time.sleep(.15)
        except FileNotFoundError:return default
    return default

def config():return read_json(HERE/'config.json')

def load_data(key):
    file=HERE/'data'/(key+'.npz')
    with np.load(file) as z:d={n:z[n] for n in z.files}
    d['length']=d['mask'].sum(1).astype(np.int32)
    src=np.flatnonzero(d['source']);target=np.flatnonzero(~d['source'])
    lo=float(d['steps'][src][d['mask'][src]].min());hi=float(d['steps'][src][d['mask'][src]].max())
    d['x']=((d['steps']-lo)/max(hi-lo,1)*2-1).astype(np.float32)
    d['condition']=np.column_stack([d['WL'],
        np.log10(d['physical'])/4 if '_Bake_' in key else d['physical']/5000,
        d['PEC']/10000]).astype(np.float32)
    d['wordlines']=256 if key.startswith('TH') else 744
    d['normalized_condition']=d['condition'].copy();d['normalized_condition'][:,0]/=d['wordlines']-1
    d['src']=src;d['target']=target;d['scale_bounds']=[lo,hi]
    return d

def fixed_test(n):
    count=min(max(2,int(round(.2*n))),n-9);chosen=[1,n-2]
    for i in np.linspace(2,n-3,count)[1:-1].round().astype(int):
        if int(i) not in chosen:chosen.append(int(i))
        if len(chosen)==count:break
    return np.array(sorted(chosen),int)

def query_order(steps,test):
    allowed=sorted(set(range(len(steps)))-set(test));chosen=[allowed[0],allowed[-1]]
    while len(chosen)<len(allowed):
        rest=[i for i in allowed if i not in chosen]
        chosen.append(max(rest,key=lambda i:(min(abs(steps[i]-steps[j]) for j in chosen),-i)))
    return np.array(chosen,int)

def split_default(n):
    interior=np.arange(1,n-1);count=max(2,round(.2*n))
    test=interior[np.linspace(0,len(interior)-1,count).round().astype(int)]
    left=np.array([i for i in interior if i not in set(test)])
    val=left[np.linspace(0,len(left)-1,count).round().astype(int)]
    train=np.array([i for i in range(n) if i not in set(test)|set(val)])
    return train,val,test

def selections(d,ids,protocol='nested',budget=7):
    fit=[];test=[];val=[]
    for i in ids:
        n=int(d['length'][i]);steps=d['steps'][i,:n]
        if protocol=='default':a,v,t=split_default(n)
        else:
            t=fixed_test(n);v=np.array([],int)
            a=query_order(steps,t)[:budget] if protocol=='nested' else np.array(sorted(set(np.linspace(0,n-1,budget).round().astype(int))-set(t)),int)
        fit.append(a);test.append(t);val.append(v)
    return fit,test,val

def padded_observations(d,ids,indices):
    m=max(len(a) for a in indices);x=np.zeros((len(ids),m),np.float32);y=x.copy();mask=x.copy()
    for j,(i,a) in enumerate(zip(ids,indices)):
        n=len(a);x[j,:n]=d['x'][i,a];y[j,:n]=d['frequency'][i,a];mask[j,:n]=1
    return x,y,mask

def score_vectors(d,ids,pred,tests,observed):
    y=d['frequency'][ids].astype(np.float64);valid=d['mask'][ids];n=d['length'][ids]
    tm=np.zeros_like(valid);tail=np.zeros_like(valid)
    for j,(length,test) in enumerate(zip(n,tests)):
        tm[j,test]=True;k=max(2,int(np.ceil(.2*length)));tail[j,:k]=True;tail[j,length-k:length]=True
    tt=tm&tail;peak=np.maximum(np.max(y,1),1e-9);err=(pred-y).astype(np.float64)
    sse=(err*err*tm).sum(1);cnt=tm.sum(1);sy=(y*tm).sum(1,dtype=np.float64);sy2=(y*y*tm).sum(1,dtype=np.float64)
    sst=sy2-sy*sy/cnt
    r2=np.divide(sse,sst,out=np.full(len(ids),np.nan),where=sst>1e-9);r2=1-r2
    rmse=lambda m:100*np.sqrt((err*err*m).sum(1)/m.sum(1))/peak
    smape=200*(np.abs(err)/np.maximum(np.abs(y)+np.abs(pred),1e-9)*tm).sum(1)/cnt
    rows=dict(task_id=d['task_id'][ids],WL=d['WL'][ids],label=d['label'][ids],physical=d['physical'][ids],PEC=d['PEC'][ids],
        test_r2=r2,test_nrmse_pct_peak=rmse(tm),test_smape_pct=smape,tail_test_rmse_pct_peak=rmse(tt),tail_full_rmse_pct_peak=rmse(tail),
        test_n=cnt,tail_test_n=tt.sum(1),observed_n=np.array([len(o) for o in observed]),sse=sse,sum_y=sy,sum_y2=sy2)
    coordinates=d['steps'][ids]
    truth_peak=y.argmax(1);pred_peak=np.where(valid,pred,-np.inf).argmax(1);rr=np.arange(len(ids))
    rows['peak_error_steps']=np.abs(coordinates[rr,truth_peak]-coordinates[rr,pred_peak])
    p=np.where(tt,y+1e-9,0);q=np.where(tt,np.maximum(pred.astype(np.float64),0)+1e-9,0);p=p/p.sum(1,keepdims=True);q=q/q.sum(1,keepdims=True)
    rows['tail_test_symmetric_kl']=.5*((p-q)*np.log(np.maximum(p,1e-300)/np.maximum(q,1e-300))).sum(1)
    def gradient(values):
        result=np.zeros_like(values,dtype=np.float64);left=coordinates[:,1:-1]-coordinates[:,:-2];right=coordinates[:,2:]-coordinates[:,1:-1]
        interior=np.arange(1,values.shape[1]-1)[None,:]<n[:,None]-1
        safe_l=np.where(interior,left,1);safe_r=np.where(interior,right,1)
        result[:,1:-1]=-safe_r/(safe_l*(safe_l+safe_r))*values[:,:-2]+(safe_r-safe_l)/(safe_l*safe_r)*values[:,1:-1]+safe_l/(safe_r*(safe_l+safe_r))*values[:,2:]
        result[:,0]=(values[:,1]-values[:,0])/(coordinates[:,1]-coordinates[:,0])
        last=n-1;result[rr,last]=(values[rr,last]-values[rr,last-1])/(coordinates[rr,last]-coordinates[rr,last-1])
        return result
    rows['tail_slope_rmse']=np.sqrt(((gradient(pred)-gradient(y))**2*tail).sum(1)/tail.sum(1))
    return rows

def parquet_write(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    frame.to_parquet(tmp,index=False,compression='zstd')
    for k in range(20):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.1*(k+1))
    raise PermissionError(f'Could not commit {path}')

def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def atomic_numpy(path,values=None,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as h:
        if values is not None:np.save(h,values,allow_pickle=False)
        else:np.savez_compressed(h,**arrays)
        h.flush();os.fsync(h.fileno())
    for k in range(20):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.1*(k+1))
    raise PermissionError(f'Could not commit array checkpoint {path}')
