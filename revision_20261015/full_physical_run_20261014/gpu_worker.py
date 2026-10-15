from pathlib import Path
import sys,time,json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from full_common import HERE,config,load_data,selections,padded_observations,score_vectors,atomic_json,parquet_write,hash_file,atomic_numpy
from gpu_ops import seed_all,new_vector,adapt,predict
from generative_models import fit_generator,generate

def source_fits(d,out,seed):
    cfg=config();ids=d['src'];file=out/'source_params.npy'
    if file.exists():return np.load(file)
    parts=out/'source_parts';parts.mkdir(exist_ok=True);start=time.perf_counter()
    for a in range(0,len(ids),cfg['gpu_batch']):
        f=parts/f'{a:07}.npy'
        if f.exists():continue
        selected=ids[a:a+cfg['gpu_batch']];init=np.repeat(new_vector(seed)[None,:],len(selected),0)
        v,info=adapt(init,d['x'][selected],d['frequency'][selected],d['mask'][selected],steps=cfg['source_steps'])
        atomic_numpy(f,v);atomic_json(parts/f'{a:07}.json',dict(rows=len(selected),nonfinite=int(info['nonfinite'].sum()),
             median_log_mse=float(np.median(info['data_loss'])),seconds=time.perf_counter()-start))
        print(f'source {a+len(selected)}/{len(ids)} fitted',flush=True)
        torch.cuda.empty_cache()
    params=np.concatenate([np.load(f) for f in sorted(parts.glob('*.npy'))]);assert len(params)==len(ids)
    atomic_numpy(file,params);return params

def init_vectors(d,params,out,seed):
    cfg=config();source=d['src'];target=d['target'];vectors={}
    # Pooled-source is a genuine shared curve regressor trained on all dense source points.
    f=out/'pooled_source.npy'
    if f.exists():pooled=np.load(f)
    else:
        valid=d['mask'][source];x=d['x'][source][valid][None,:];y=d['frequency'][source][valid][None,:]
        pooled=adapt(new_vector(seed)[None,:],x,y,np.ones_like(x),steps=cfg['source_steps'])[0][0];atomic_numpy(f,pooled)
    vectors['pooled_source']=np.repeat(pooled[None,:],len(target),0)
    cx=d['normalized_condition'][source];ct=d['normalized_condition'][target]
    cm=cx.mean(0);cs=np.maximum(cx.std(0),1e-3)
    ridge=Ridge(alpha=1).fit((cx-cm)/cs,params);vectors['conditional_ridge']=ridge.predict((ct-cm)/cs).astype(np.float32)
    # Nearest condition without allocating target x source x feature tensors.
    sx=(cx-cm)/cs;tx=(ct-cm)/cs;nearest=[]
    from scipy.spatial import cKDTree
    tree=cKDTree(sx);_,indices=tree.query(tx,k=1);vectors['nearest_source']=params[indices]
    vectors['scratch']=np.stack([new_vector(seed+int(d['task_id'][i])) for i in target])
    return vectors

def model_predictions(d,out,seed,method,initial,protocols=None,strength=0,optimizer='lbfgs'):
    cfg=config();target=d['target'];base=out/'metrics'/method
    protocols=protocols or [('nested',b) for b in cfg['budgets']]+[('default',0)]
    for protocol,B in protocols:
      for a in range(0,len(target),cfg['gpu_batch']):
        file=base/f'{protocol}_B{B}_{a:07}.parquet'
        pred_file=out/'predictions'/method/f'{protocol}_B{B}_{a:07}.npz'
        default_file=out/'default_params'/method/f'{a:07}.npz'
        if file.exists() and pred_file.exists() and (protocol!='default' or default_file.exists()):continue
        ids=target[a:a+cfg['gpu_batch']];fit,test,val=selections(d,ids,protocol,B)
        x,y,m=padded_observations(d,ids,fit)
        validation=padded_observations(d,ids,val) if protocol=='default' else None
        epochs=250 if optimizer=='adam' else cfg['target_steps'];lr=.01 if optimizer=='adam' else cfg['lbfgs_lr']
        v,info=adapt(initial[a:a+len(ids)],x,y,m,steps=epochs,lr=lr,strength=strength,optimizer=optimizer,validation=validation)
        prediction=predict(v,d['x'][ids]);row=pd.DataFrame(score_vectors(d,ids,prediction,test,fit))
        row['method']=method;row['seed']=seed;row['protocol']=protocol;row['budget']=B;row['lambda']=strength
        row['validation_n']=np.array([len(v) for v in val]);row['measured_n']=row['observed_n']+row['validation_n']
        row['nonfinite']=info['nonfinite'];row['data_loss']=info['data_loss'];row['penalty_loss']=info['penalty_loss']
        row['selection_loss']=info['monitor']
        pred_dir=out/'predictions'/method;pred_dir.mkdir(parents=True,exist_ok=True)
        atomic_numpy(pred_file,pred=prediction,task_id=d['task_id'][ids],length=d['length'][ids])
        if protocol=='default':
            # Candidate selection is source of extra validation measurements, never test frequencies.
            params_dir=out/'default_params'/method;params_dir.mkdir(parents=True,exist_ok=True)
            atomic_numpy(default_file,params=v,validation=info['monitor'])
        parquet_write(file,row)
        atomic_json(out/'progress.json',dict(status='running',method=method,protocol=protocol,budget=B,completed_through=a+len(ids),
                target_curves=len(target),updated=time.time(),full_target_coverage=True))
        print(f'{method} {protocol} B{B}: {a+len(ids)}/{len(target)}',flush=True)

def generated_initial(d,params,out,seed,variant,subset=None,force_updates=None):
    f=out/'generated'/f'{variant}.npy';f.parent.mkdir(exist_ok=True)
    if f.exists():return np.load(f)
    source=d['src'] if subset is None else d['src'][subset]
    training=params if subset is None else params[subset]
    net,record=fit_generator(training,d['condition'][source],d['wordlines'],seed,variant,out/f'generator_{variant}.pt',force_updates=force_updates)
    values,seconds=generate(net,record,d['condition'][d['target']],seed,variant)
    atomic_numpy(f,values)
    atomic_json(out/'generated'/f'{variant}.json',dict(rows=len(values),seconds=seconds,source_vectors=len(training),seed=seed,
        checksum=hash_file(f),variant=variant,epochs=record['epochs'],updates=record['optimizer_updates'],min_source_exposure=record['min_source_exposure']))
    del net;torch.cuda.empty_cache();return values

def selection_policy(d,out,seed):
    folder=out/'metrics/selected_policy';folder.mkdir(parents=True,exist_ok=True)
    target=d['target'];cfg=config()
    for a in range(0,len(target),cfg['gpu_batch']):
        file=folder/f'default_B0_{a:07}.parquet'
        if file.exists():continue
        gen=np.load(out/f'default_params/diffusion/{a:07}.npz');scratch=np.load(out/f'default_params/scratch/{a:07}.npz')
        selected=gen['validation']<=scratch['validation'];params=np.where(selected[:,None],gen['params'],scratch['params'])
        ids=target[a:a+len(params)];fit,test,val=selections(d,ids,'default',0)
        row=pd.DataFrame(score_vectors(d,ids,predict(params,d['x'][ids]),test,fit));row['method']='selected_policy';row['seed']=seed
        row['protocol']='default';row['budget']=0;row['generated_selected']=selected;parquet_write(file,row)

def main(key,seed,stage):
    assert torch.cuda.is_available();seed_all(seed);d=load_data(key);out=HERE/'results'/key/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    done=out/f'{stage}_complete.json'
    if done.exists():return
    params=source_fits(d,out,seed)
    if stage=='main':
        values=generated_initial(d,params,out,seed,'main')
        vectors=init_vectors(d,params,out,seed);vectors['diffusion']=values
        for name,initial in vectors.items():model_predictions(d,out,seed,name,initial)
        selection_policy(d,out,seed)
        atomic_json(done,dict(status='complete',target_curves=len(d['target']),source_curves=len(d['src']),seed=seed,
                       methods=list(vectors)+['selected_policy'],all_target_curves=True))
    elif stage=='regularization':
        # Dedicated marker: a lambda-only job must never masquerade as the old full ablation.
        base=generated_initial(d,params,out,seed,'main')
        for strength in cfg_values('lambda_values'):
            if strength==0:continue
            model_predictions(d,out,seed,f'lambda_{strength:g}',base,[('nested',7)],strength=strength)
        atomic_json(done,dict(status='complete',scope='retained_reviews_20261014',seed=seed,
            full_targets=len(d['target']),lambda_values=cfg_values('lambda_values'),protocol='nested',budget=7,
            lambda_zero_reference='metrics/diffusion/nested_B7',all_target_curves=True,
            interpretation='fixed sensitivity grid, not test-selected default'))
    elif stage=='ablation':
        base=generated_initial(d,params,out,seed,'main')
        for strength in cfg_values('lambda_values'):
            if strength==0:continue
            model_predictions(d,out,seed,f'lambda_{strength:g}',base,[('nested',7)],strength=strength)
        for variant in ['T50_native','T50_matched','noWL','noStress','noPEC','unconditional','cvae','flow','dropout']:
            initial=generated_initial(d,params,out,seed,variant)
            model_predictions(d,out,seed,variant,initial,[('nested',7)])
        order=np.random.default_rng(42).permutation(len(params))
        for number in [50,150,len(params)//2]:
            subset=np.sort(order[:number]);name=f'source{number}'
            initial=generated_initial(d,params,out,seed,name,subset)
            model_predictions(d,out,seed,name,initial,[('nested',7)])
        # Matched-noise condition shuffling on the full target universe.
        net,record=fit_generator(params,d['condition'][d['src']],d['wordlines'],seed,'main',out/'generator_main.pt')
        condition=d['condition'][d['target']]
        for col,label in enumerate(['WL','Stress','PEC']):
            name=f'shuffle_{label}';f=out/'generated'/f'{name}.npy'
            if f.exists():values=np.load(f)
            else:
                changed=condition.copy();idx=np.random.default_rng(20261014+col).permutation(len(changed));changed[:,col]=changed[idx,col]
                values,_=generate(net,record,changed,seed,'main');atomic_numpy(f,values)
                sd=np.maximum(params.std(0),1e-3)
                distance=float(np.mean(np.abs(np.sort(values/sd,axis=0)-np.sort(base/sd,axis=0))))
                atomic_json(out/'generated'/f'{name}.json',dict(source_standardized_marginal_W1=distance,matched_noise=True,full_targets=len(values)))
            model_predictions(d,out,seed,name,values,[('nested',7)])
        del net;torch.cuda.empty_cache()
        for name,initial in [('diffusion',base)]+list(init_vectors(d,params,out,seed).items()):
            if name in ('diffusion','nearest_source','pooled_source'):model_predictions(d,out,seed,f'{name}_adam250',initial,[('nested',7)],optimizer='adam')
        # Deep ensemble consists of independently initialized source conditional means, not copied fits.
        ensemble=[]
        for member in range(3):
            v=generated_initial(d,params,out,seed+1000*(member+1),f'ensemble_member{member}')
            ensemble.append(v)
        model_predictions(d,out,seed,'ensemble',np.mean(ensemble,axis=0),[('nested',7)])
        atomic_json(done,dict(status='complete',full_targets=len(d['target']),source_size_ablation='generator source set only',
                      source_exposure_control='two full passes; optimizer-update counts are recorded and not called matched compute'))

def cfg_values(name):return config()[name]

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),sys.argv[3])
